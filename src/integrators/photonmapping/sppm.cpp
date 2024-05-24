#include <mutex>
#include <list>
#include <shared_mutex>
#include <fstream>

#include <drjit/morton.h>
#include <drjit/dynamic.h>

#include <mitsuba/core/atomic.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <nanothread/nanothread.h>

NAMESPACE_BEGIN(mitsuba)

//======================================
#define PBRT_L1_CACHE_LINE_SIZE 64

template <typename T>
struct AllocationTraits
{
    using SingleObject = T *;
};
template <typename T>
struct AllocationTraits<T[]>
{
    using Array = T *;
};
template <typename T, size_t n>
struct AllocationTraits<T[n]>
{
    struct Invalid
    {
    };
};

// ScratchBuffer Definition
// alignas helps to prevent false sharing cache misses
// https://youtu.be/BP6NxVxDQIs?si=eBacfOW4zYwV5Xpo&t=2889
class alignas(PBRT_L1_CACHE_LINE_SIZE) ScratchBuffer
{
public:
    // ScratchBuffer Public Methods
    ScratchBuffer(int size = 256) : allocSize(size)
    {
        // ptr = (char *)Allocator().allocate_bytes(size, align);
        ptr = (char *)new uint8_t[size];
    }

    ScratchBuffer(const ScratchBuffer &) = delete;

    ScratchBuffer(ScratchBuffer &&b)
    {
        ptr = b.ptr;
        allocSize = b.allocSize;
        offset = b.offset;
        smallBuffers = std::move(b.smallBuffers);

        b.ptr = nullptr;
        b.allocSize = b.offset = 0;
    }

    ~ScratchBuffer()
    {
        Reset();
        // Allocator().deallocate_bytes(ptr, allocSize, align);
        delete[] ptr;
    }

    ScratchBuffer &operator=(const ScratchBuffer &) = delete;

    ScratchBuffer &operator=(ScratchBuffer &&b)
    {
        std::swap(b.ptr, ptr);
        std::swap(b.allocSize, allocSize);
        std::swap(b.offset, offset);
        std::swap(b.smallBuffers, smallBuffers);
        return *this;
    }

    void *Alloc(size_t size, size_t align)
    {
        if ((offset % align) != 0)
            offset += align - (offset % align);
        if (offset + size > allocSize)
            Realloc(size);
        void *p = ptr + offset;
        offset += size;
        return p;
    }

    template <typename T, typename... Args>
    typename AllocationTraits<T>::SingleObject Alloc(Args &&...args)
    {
        T *p = (T *)Alloc(sizeof(T), alignof(T));
        return new (p) T(std::forward<Args>(args)...);
    }

    template <typename T>
    typename AllocationTraits<T>::Array Alloc(size_t n = 1)
    {
        using ElementType = typename std::remove_extent_t<T>;
        ElementType *ret =
            (ElementType *)Alloc(n * sizeof(ElementType), alignof(ElementType));
        for (size_t i = 0; i < n; ++i)
            new (&ret[i]) ElementType();
        return ret;
    }

    void Reset()
    {
        for (const auto &buf : smallBuffers)
        {
            // Allocator().deallocate_bytes(buf.first, buf.second, align);
            delete[] buf.first;
        }
        smallBuffers.clear();
        offset = 0;
    }

private:
    // ScratchBuffer Private Methods
    void Realloc(size_t minSize)
    {
        smallBuffers.push_back(std::make_pair(ptr, allocSize));
        allocSize = std::max(2 * minSize, allocSize + minSize);
        // ptr = (char *)Allocator().allocate_bytes(allocSize, align);
        ptr = (char *)new uint8_t[allocSize];
        offset = 0;
    }

    // ScratchBuffer Private Members
    static constexpr int align = PBRT_L1_CACHE_LINE_SIZE;
    char *ptr = nullptr;
    int allocSize = 0, offset = 0;
    std::list<std::pair<char *, size_t>> smallBuffers;
};

int RunningThreads()
{
    return Thread::thread()->thread_count();
}
// =====================================
// ThreadLocal Definition
template <typename T>
class ThreadLocal
{
public:
    // ThreadLocal Public Methods
    ThreadLocal() : hashTable(4 * RunningThreads()), create([]()
                                                            { return T(); }) {}
    ThreadLocal(std::function<T(void)> &&c)
        : hashTable(4 * RunningThreads()), create(c) {}

    T &Get();

    template <typename F>
    void ForAll(F &&func);

private:
    // ThreadLocal Private Members
    struct Entry
    {
        uint32_t tid;
        T value;
    };
    std::shared_mutex mutex;
    std::vector<std::optional<Entry>> hashTable;
    std::function<T(void)> create;
};

#define CHECK_IMPL(a, b, op) assert((a)op(b))
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)

// ThreadLocal Inline Methods
template <typename T>
inline T &ThreadLocal<T>::Get()
{
    // std::thread::id tid = std::this_thread::get_id();
    uint32_t tid = Thread::thread()->thread_id();
    uint32_t hash = std::hash<uint32_t>()(tid);
    hash %= hashTable.size();
    int step = 1;
    int tries = 0;

    mutex.lock_shared();
    while (true)
    {
        CHECK_LT(++tries, hashTable.size()); // full hash table

        if (hashTable[hash] && hashTable[hash]->tid == tid)
        {
            // Found it
            T &threadLocal = hashTable[hash]->value;
            mutex.unlock_shared();
            return threadLocal;
        }
        else if (!hashTable[hash])
        {
            mutex.unlock_shared();

            // Get reader-writer lock before calling the callback so that the user
            // doesn't have to worry about writing a thread-safe callback.
            mutex.lock();
            T newItem = create();

            if (hashTable[hash])
            {
                // someone else got there first--keep looking, but now
                // with a writer lock.
                while (true)
                {
                    hash += step;
                    ++step;
                    if (hash >= hashTable.size())
                        hash %= hashTable.size();

                    if (!hashTable[hash])
                        break;
                }
            }

            hashTable[hash] = Entry{tid, std::move(newItem)};
            T &threadLocal = hashTable[hash]->value;
            mutex.unlock();
            return threadLocal;
        }

        hash += step;
        ++step;
        if (hash >= hashTable.size())
            hash %= hashTable.size();
    }
}

template <typename T>
template <typename F>
inline void ThreadLocal<T>::ForAll(F &&func)
{
    mutex.lock();
    for (auto &entry : hashTable)
    {
        if (entry)
            func(entry->value);
    }
    mutex.unlock();
}
// ================================================

template <typename Float>
inline uint64_t Hash(const Point<dr::int32_array_t<Float>, 3> &p)
{
    return ((p.x() * 73856093) ^ (p.y() * 19349663) ^ (p.z() * 83492791));
}

// SPPMPixel
template <typename Float, typename Spectrum>
struct VisiblePoint
{

    MI_IMPORT_TYPES(BSDFPtr)

    Point3f p;
    Vector3f wo;
    BSDFPtr bsdf;
    SurfaceInteraction3f si;
    Spectrum beta;
    Bool secondaryLambdaTerminated;
    VisiblePoint(const SurfaceInteraction3f &si, const Point3f &p, const Vector3f &wo, const BSDFPtr &bsdf, const Spectrum &beta, bool secondaryLambdaTerminated)
        : si(si), p(p), wo(wo), bsdf(bsdf), beta(beta), secondaryLambdaTerminated(secondaryLambdaTerminated) {}

    DRJIT_STRUCT(VisiblePoint, si, p, wo, bsdf, beta, secondaryLambdaTerminated)
};

template <typename Float, typename Spectrum>
struct SPPMPixel
{

    MI_IMPORT_TYPES()
    using VisiblePoint = VisiblePoint<Float, Spectrum>;

    Float radius = 0.f;
    Color3f Ld = 0.f;

    VisiblePoint vp;

    AtomicFloat<Float> Phi_i[3];
    std::atomic<UInt32> M;
    Color3f tau = 0.f;
    Float n = 0;

    DRJIT_STRUCT(SPPMPixel, radius, Ld, vp, Phi_i, M, tau, n)
};

template <typename Float, typename Spectrum>
struct SPPMPixelListNode
{
    SPPMPixel<Float, Spectrum> *pixel;
    SPPMPixelListNode *next;
};

template <typename Float, typename Spectrum>
class SPPMIntegrator : public Integrator<Float, Spectrum>
{
public:
    MI_IMPORT_BASE(Integrator, aov_names, should_stop, m_render_timer, m_stop)
    MI_IMPORT_TYPES(Scene, Sensor, ImageBlock, Sampler, Film, MediumPtr, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    using SPPMPixel = SPPMPixel<Float, Spectrum>;
    using BoundingBox2u = BoundingBox<Point2u>;
    using SPPMPixelListNode = SPPMPixelListNode<Float, Spectrum>;

    SPPMIntegrator(const Properties &props) : Base(props)
    {
        m_block_size = props.get<uint32_t>("block_size", 0);
        m_max_depth = props.get<uint32_t>("max_depth", 5);
        m_rr_depth = props.get<uint32_t>("rr_depth", 5);
        m_initial_radius = props.get<float>("initial_radius", 0.f);
    }

    std::pair<Ray3f, Spectrum> prepare_ray(const Scene *scene,
                                           const Sensor *sensor,
                                           Sampler *sampler) const
    {
        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0)
            time += sampler->next_1d() * sensor->shutter_open_time();

        // Prepare random samples.
        Float wavelength_sample = sampler->next_1d();
        Point2f direction_sample = sampler->next_2d(),
                position_sample = sampler->next_2d();

        // Sample one ray from an emitter in the scene.
        auto [ray, ray_weight, emitter] = scene->sample_emitter_ray(
            time, wavelength_sample, direction_sample, position_sample);

        return {ray, ray_weight};
    }

    TensorXf render(Scene *scene,
                    Sensor *sensor,
                    uint32_t seed = 0,
                    uint32_t spp = 0,
                    bool develop = true,
                    bool evaluate = true) override
    {

        // Final result output
        TensorXf result;

        Log(Info, "scene bounds: %s", scene->bbox());

        auto bbox = scene->bbox();
        ScalarFloat scene_radius = dr::norm(bbox.max - bbox.min) / 2;
        Film *film = sensor->film();
        ScalarVector2u film_size = film->crop_size();

        if (m_initial_radius == 0.f)
        {
            m_initial_radius = 5 * std::min(scene_radius / film_size.x(), scene_radius / film_size.y());
        }

        // implement only for cpu type
        if constexpr (!dr::is_jit_v<Float>)
        {
            // Potentially adjust the number of samples per pixel if spp != 0
            Sampler *sampler = sensor->sampler();
            if (spp)
                sampler->set_sample_count(spp);
            spp = sampler->sample_count();

            Log(Info, "spp: %u", spp);
            uint32_t num_iter = spp;

            Film *film = sensor->film();
            ScalarVector2u film_size = film->crop_size();
            if (film->sample_border())
                film_size += 2 * film->rfilter()->border_size();

            // Determine output channels and prepare the film with this information
            // needed for developing default film
            size_t n_channels = film->prepare(aov_names());

            uint32_t nPixels = film_size.x() * film_size.y();

            // Render on the CPU using a spiral pattern
            // generic parallelization inits
            uint32_t n_threads = (uint32_t)Thread::thread_count();
            // If no block size was specified, find size that is good for parallelization
            uint32_t block_size = m_block_size;
            if (block_size == 0)
            {
                block_size = MI_BLOCK_SIZE; // 32x32
                while (true)
                {
                    // Ensure that there is a block for every thread
                    if (block_size == 1 || dr::prod((film_size + block_size - 1) /
                                                    block_size) >= n_threads)
                        break;
                    block_size /= 2;
                }
            }

            // SPPM pixels

            // Create a 2D grid of SPPM pixels
            // dr::DynamicArray<SPPMPixel> pixels;
            // dr::set_slices(pixels, nPixels);
            // unclear how to allocate memory for SPPMPixel vector
            std::vector<SPPMPixel> pixels(nPixels);

            // assign radius
            for (uint32_t i = 0; i < nPixels; ++i)
            {
                // pixels[i].radius = m_initial_radius;
                // dr::slice(pixels, i).radius = m_initial_radius;
                pixels.at(i).radius = m_initial_radius;
            }

            std::mutex mutex;
            ref<ProgressReporter> progress_visible, progress_photon;
            Logger *logger = mitsuba::Thread::thread()->logger();
            if (logger && Info >= logger->log_level())
            {
                progress_visible = new ProgressReporter("Stage1: Visible Points");
                progress_photon = new ProgressReporter("Stage3: Photon Tracing");
            }

            Log(Info, "About to start the main loop for %u", num_iter);
            std::vector<ScalarFloat> Ld_data_(nPixels * 3);
            // run main loop
            for (uint32_t iter = 0; iter < num_iter; ++iter)
            {
                Log(Info, "Starting iteration %u/%u", iter + 1, num_iter);

                Spiral spiral(film_size, film->crop_offset(), block_size, 1);
                // Total number of blocks to be handled, including multiple passes.
                uint32_t total_blocks = spiral.block_count() * 1,
                         blocks_done = 0;

                // Grain size for parallelization
                uint32_t grain_size = std::max(total_blocks / (4 * n_threads), 1u);

                Log(Info, "Starting visible point render job (%ux%u)",
                    film_size.x(), film_size.y());
                progress_visible->update(0);
                // Generate SPPM visible points
                ThreadEnvironment env;
                dr::parallel_for(
                    dr::blocked_range<uint32_t>(0, total_blocks, grain_size),
                    [&](const dr::blocked_range<uint32_t> &range)
                    {
                        ScopedSetThreadEnvironment set_env(env);

                        // Fork a non-overlapping sampler for the current worker
                        ref<Sampler> local_sampler = sensor->sampler()->fork();

                        // Render up to 'grain_size' image blocks
                        for (uint32_t i = range.begin();
                             i != range.end() && !should_stop(); ++i)
                        {

                            auto [offset, size, block_id] = spiral.next_block();
                            Assert(dr::prod(size) != 0);

                            Log(Debug, "Rendering block %u/%u (offset: %s, size: %s)",
                                i + 1, total_blocks, offset, size);

                            // TODO: check if the following should be uncommented
                            // if (film->sample_border())
                            //     offset -= film->rfilter()->border_size();

                            Vector2i spiral_block_size = size;
                            Vector2u spiral_block_offset = offset;

                            // seeding is done in the render from camera
                            // main path tracing style step
                            render_from_camera(scene, sensor, local_sampler, pixels,
                                               film_size, spiral_block_size, spiral_block_offset,
                                               seed, block_id, block_size, iter, num_iter);

                            // not necessary as render_from_camera has to deal with it
                            // update sampler state
                            // sampler->advance();

                            /* Critical section: update progress bar */
                            if (progress_visible)
                            {
                                std::lock_guard<std::mutex> lock(mutex);
                                blocks_done++;
                                progress_visible->update(blocks_done / (Float)total_blocks);
                            }

                        } // for loop end
                    } // lambda end
                ); // parallel_for end

                // Allocate grid for SPPM visible points
                uint32_t hashSize = nPixels;
                std::vector<std::atomic<SPPMPixelListNode *>> grid(hashSize);
                std::vector<std::atomic<int>> gridCounter(hashSize);

                // Compute grid bounds for SPPM visible points
                BoundingBox3f gridBounds;
                Float maxRadius = 0.f;
                for (const auto &pixel : pixels)
                {
                    if (!dr::mean(pixel.vp.beta))
                        continue;

                    auto vpBounds = BoundingBox3f(
                        pixel.vp.p - Vector3f(pixel.radius),
                        pixel.vp.p + Vector3f(pixel.radius));

                    // Log(Debug, "Visible point p: %s, beta: %s", pixel.vp.p, pixel.vp.beta);

                    gridBounds.expand(vpBounds);
                    maxRadius = dr::maximum(maxRadius, pixel.radius);
                }

                // Allocate per-thread _ScratchBuffer_s for SPPM rendering
                ThreadLocal<ScratchBuffer> threadScratchBuffers(
                    []()
                    { return ScratchBuffer(1024 * 1024); });

                // Compute resolution of SPPM grid in each dimension
                // UInt32 gridRes[3];
                Vector3f diag = gridBounds.extents();
                Float maxDiag = dr::max(diag);
                int baseGridRes = int(maxDiag / maxRadius);
                Point3i gridRes = dr::maximum(1, dr::floor2int<Vector3f>(baseGridRes * diag / maxDiag));

                Log(Debug, "Hashing %u visible points into a %s, gridBounds: %s",
                    nPixels, gridRes, gridBounds);

                // Add visible points to SPPM grid
                // iterate over pixels
                // run over rows first
                //         dr::parallel_for(
                //             dr::blocked_range<uint32_t>(0, nPixels, n_threads),
                //             [&](const dr::blocked_range<uint32_t> &range){
                //                 ScopedSetThreadEnvironment set_env(env);

                //                 // need to get a local arena which is disjoint
                //                 const uint32_t thread_id = Thread::thread()->thread_id();
                //                 // Log(Info, "Creating scratchbuffer for : %u",
                //                 //     thread_id);
                //                 // auto &scratchBuffer = *threadScratchBuffer[thread_id];
                //                 ScratchBuffer &scratchBuffer = threadScratchBuffers.Get();
                //                 // ----

                //                 for (uint32_t i = range.begin();
                //                     i != range.end() &&!should_stop(); ++i) {

                //                     uint32_t px_id = i;
                //                     Point2u block_px_start(px_id % film_size.x(), px_id / film_size.x());

                //                     SPPMPixel &pixel = pixels.at(px_id);
                // //                    Log(Debug, "Adding pixel with pos: %s", pixel.vp.p);
                // //                    if (pixel.vp.p.x() != pixel.vp.p.x()) {
                // //                        Log(Debug, "------- something is wrong: %s", pixel.vp.p);
                // //                        Point po = pixel.vp.p;
                // //                    }
                //                     if ( !(pixel.vp.p.x() != pixel.vp.p.x()) && dr::mean(pixel.vp.beta) ) {
                //                         // Add pixel's visible point to applicable grid cells
                //                         // Find grid cell bounds for pixel's visible point, _pMin_ and _pMax_
                //                         Float pixelRadius = pixel.radius;
                //                         // Point3i pMin, pMax;
                //                         // find the grid cell where vp - r and vp + r are located
                //                         Point3f vp_min = pixel.vp.p - Vector3f(pixelRadius),
                //                             vp_max = pixel.vp.p + Vector3f(pixelRadius);

                //                         vp_min = (vp_min - gridBounds.min) / gridBounds.extents();
                //                         vp_max = (vp_max - gridBounds.min) / gridBounds.extents();

                //                         Point3i pMin = dr::clamp(
                //                             dr::floor2int<Point3i>(gridRes * vp_min),
                //                             Point3i(0), gridRes);

                //                         Point3i pMax = dr::clamp(
                //                             dr::floor2int<Point3i>(gridRes * vp_max),
                //                             Point3i(0), gridRes);

                //                         // loop over all the grid cells and splat by looping over
                //                         for (int z = pMin.z(); z <= pMax.z(); ++z) {
                //                             for (int y = pMin.y(); y <= pMax.y(); ++y) {
                //                                 for (int x = pMin.x(); x <= pMax.x(); ++x) {
                //                                     // Add visible point to grid cell $(x, y, z)$
                //                                     int h = Hash<Float>(Point3i(x, y, z)) % hashSize;
                //                                     // allocate nodes using global memory so that they don't vanish when the loop ends
                //                                     SPPMPixelListNode *node = scratchBuffer.Alloc<SPPMPixelListNode>();
                //                                     // linked list
                //                                     Log(Debug, "Adding pixel with pos: %s", pixel.vp.p);
                //                                     if (pixel.vp.p.x() != pixel.vp.p.x()) {
                //                                         Log(Debug, "------- something is wrong: %s", pixel.vp.p);
                //                                         Point po = pixel.vp.p;
                //                                     }
                //                                     node->pixel = &pixel;

                //                                     // change the grid node counter
                //                                     gridCounter[h]++;
                //                                     // add an entry to linked list in the SPPMNode
                //                                     node->next = grid[h].load(std::memory_order_relaxed);
                //                                     while (!grid[h].compare_exchange_weak(node->next, node))
                //                                         ;
                //                                 }
                //                             }
                //                         }

                //                     }

                //                 }
                //             }
                //         );

                // ---
                // write a serial version of grid splatting
                ScratchBuffer &myScratchBuffer = threadScratchBuffers.Get();
                for (uint32_t px_id = 0; px_id < nPixels; px_id++)
                {
                    SPPMPixel &pixel = pixels.at(px_id);
                    if (!(pixel.vp.p.x() != pixel.vp.p.x()) && dr::mean(pixel.vp.beta))
                    {
                        // Add pixel's visible point to applicable grid cells
                        // Find grid cell bounds for pixel's visible point, _pMin_ and _pMax_
                        Float pixelRadius = pixel.radius;
                        // find the grid cell where vp - r and vp + r are located
                        Point3f vp_min = pixel.vp.p - Vector3f(pixelRadius),
                                vp_max = pixel.vp.p + Vector3f(pixelRadius);

                        vp_min = (vp_min - gridBounds.min) / gridBounds.extents();
                        vp_max = (vp_max - gridBounds.min) / gridBounds.extents();

                        Point3i pMin = dr::clamp(
                            dr::floor2int<Point3i>(gridRes * vp_min),
                            Point3i(0), gridRes);

                        Point3i pMax = dr::clamp(
                            dr::floor2int<Point3i>(gridRes * vp_max),
                            Point3i(0), gridRes);

                        // loop over all the grid cells and splat by looping over
                        for (int zz = pMin.z(); zz <= pMax.z(); ++zz)
                        {
                            for (int yy = pMin.y(); yy <= pMax.y(); ++yy)
                            {
                                for (int xx = pMin.x(); xx <= pMax.x(); ++xx)
                                {
                                    // DONOT remove: the following code does some magic and sppm works because of it
                                    // Log(Trace, "magic line %u", xx);
                                    // =====
                                    // Add visible point to grid cell $(x, y, z)$
                                    const Point3i grid_p(xx, yy, zz);
                                    int hh = Hash<Float>(grid_p) % hashSize;
                                    // allocate nodes using global memory so that they don't vanish when the loop ends
                                    SPPMPixelListNode *node = myScratchBuffer.Alloc<SPPMPixelListNode>();
                                    // linked list
                                    node->pixel = &pixel;
                                    // change the grid node counter
                                    gridCounter[hh]++;
                                    // add an entry to linked list in the SPPMNode
                                    node->next = grid[hh].load(std::memory_order_relaxed);
                                    while (!grid[hh].compare_exchange_weak(node->next, node))
                                    {
                                    }
                                } // end xx
                            } // end yy
                        } // end zz
                    } // end pixel check if
                } // end of pixel for loop

                int photonsPerIteration = nPixels;

                size_t n_threads = Thread::thread_count();
                size_t grain_size_photon = std::max(photonsPerIteration / (4 * n_threads), (size_t)1);

                std::atomic<size_t> samples_done(0);

                Log(Debug, "grain_size: %u, n_thread: %u",
                    grain_size_photon, n_threads);

                Log(Info, "Starting photon tracing pass, with %u photons", photonsPerIteration);

                // Trace photons and accumulate contributions
                dr::parallel_for(
                    dr::blocked_range<size_t>(0, photonsPerIteration, grain_size_photon),
                    [&](const dr::blocked_range<size_t> &range)
                    {
                        ScopedSetThreadEnvironment set_env(env);

                        // Fork a non-overlapping sampler for the current worker
                        ref<Sampler> local_sampler = sensor->sampler()->clone();

                        // this will not account for iter
                        // sampler->seed(seed +
                        //     (uint32_t) range.begin() / (uint32_t) grain_size_photon);

                        Log(Debug, "Photon tracing from %u to %u", range.begin(), range.end());

                        // process up to 'grain_size' image blocks
                        for (size_t photonIndex = range.begin();
                             photonIndex != range.end() && !should_stop(); ++photonIndex)
                        {

                            local_sampler->seed(seed +
                                                (uint32_t)iter * (uint32_t)photonsPerIteration + photonIndex);

                            // generate photon rays
                            auto [ray, throughput] = prepare_ray(scene, sensor, local_sampler);
                            if (dr::all(throughput == 0.f))
                                continue;

                            // Follow photon path through scene and record intersections
                            Float eta(1.f);
                            Int32 depth = 1;
                            Mask active = true;
                            // First intersection from the emitter to the scene
                            SurfaceInteraction3f si = scene->ray_intersect(ray, active);
                            active &= si.is_valid();
                            if (m_max_depth >= 0)
                                active &= depth < m_max_depth;

                            dr::Loop<Mask> loop("Photon Tracer", active, depth, ray,
                                                throughput, si, eta, sampler);

                            while (loop(active))
                            {
                                // contribute to visible points
                                Point3i photonGridIndex = Point3i(
                                    dr::floor2int<Point3i>(gridRes * (si.p - gridBounds.min) / gridBounds.extents()));

                                // check inbounds
                                if (dr::all(photonGridIndex >= 0 && photonGridIndex < gridRes))
                                {
                                    // Compute the hash value for the grid cell
                                    int h = Hash<Float>(photonGridIndex) % hashSize;
                                    const int PhotonsInList = gridCounter[h].load();

                                    int photon_id = 0;
                                    // node is not a nullptr even at the end; Maybe this is causing nodes into a invalid state
                                    for (SPPMPixelListNode *node =
                                             grid[h].load(std::memory_order_relaxed);
                                         node; node = node->next)
                                    {

                                        if (photon_id >= PhotonsInList)
                                            Log(Debug, "->->->->->-> PI:%u D:%u, visible_id: %u/%u, next(Null):%u",
                                                photonIndex, depth, photon_id, PhotonsInList, node->next == nullptr);

                                        photon_id++;
                                        SPPMPixel &pixel = *node->pixel;
                                        // not a neighbor
                                        if (dr::squared_norm(pixel.vp.p - si.p) > dr::sqr(pixel.radius))
                                            continue;

                                        // Update pixel Phi and M for nearby photon
                                        Vector3f wi = -ray.d;
                                        BSDFContext ctx;
                                        Vector3f wo_local = pixel.vp.si.to_local(wi);
                                        Spectrum Phi = throughput * pixel.vp.bsdf->eval(ctx, pixel.vp.si, wo_local);

                                        // TODO: account for sampled wavelengths
                                        Spectrum Phi_i = pixel.vp.beta * Phi;
                                        // atomic add
                                        for (int i = 0; i < 3; ++i)
                                            pixel.Phi_i[i] += Phi_i[i];

                                        ++pixel.M;
                                    } // iteration over linked list end

                                } // inbounds check

                                // continue tracing or decide to close some off
                                /* ----------------------- BSDF sampling ------------------------ */
                                // Sample BSDF * cos(theta).
                                BSDFPtr bsdf = si.bsdf(ray);
                                BSDFContext ctx(TransportMode::Importance);
                                auto [bs, bsdf_val] =
                                    bsdf->sample(ctx, si, local_sampler->next_1d(active),
                                                 local_sampler->next_2d(active), active);

                                // Using geometric normals (wo points to the camera)
                                Float wi_dot_geo_n = dr::dot(si.n, -ray.d),
                                      wo_dot_geo_n = dr::dot(si.n, si.to_world(bs.wo));

                                // Prevent light leaks due to shading normals
                                active &= (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                                          (wo_dot_geo_n * Frame3f::cos_theta(bs.wo) > 0.f);

                                // Adjoint BSDF for shading normals -- [Veach, p. 155]
                                Float correction = dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                                                           (Frame3f::cos_theta(bs.wo) * wi_dot_geo_n));
                                throughput *= bsdf_val * correction;
                                eta *= bs.eta;

                                active &= dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
                                if (dr::none_or<false>(active))
                                    break;

                                // Intersect the BSDF ray against scene geometry (next vertex).
                                ray = si.spawn_ray(si.to_world(bs.wo));
                                si = scene->ray_intersect(ray, active);

                                depth++;
                                if (m_max_depth >= 0)
                                    active &= depth < m_max_depth;
                                active &= si.is_valid();
                                // -----

                                // Russian Roulette
                                Mask use_rr = depth > m_rr_depth;
                                if (dr::any_or<true>(use_rr))
                                {
                                    Float q = dr::minimum(
                                        dr::max(unpolarized_spectrum(throughput)) * dr::sqr(eta), 0.95f);
                                    dr::masked(active, use_rr) &= local_sampler->next_1d(active) < q;
                                    dr::masked(throughput, use_rr) *= dr::rcp(q);
                                }

                            } // tracing loop

                            local_sampler->advance();

                        } // while loop end

                        samples_done += (range.end() - range.begin());

                        // locked
                        std::lock_guard<std::mutex> lock(mutex);
                        progress_photon->update(samples_done / (ScalarFloat)photonsPerIteration);
                    });

                progress_photon->update(0.0);
                // reset scratch space after tracing photons
                // threadScratchBuffers.clear();
                // for(auto& thread_id: threadScratchBuffer) {
                //     threadScratchBuffer[thread_id].release();
                // }
                // threadScratchBuffers.ForAll([](ScratchBuffer &buffer) {
                //     // TODO: free all chunks
                //     buffer.Reset();
                // });

                // update counters
                // photonPaths += photonsPerIteration;

                uint32_t n_threads_ = Thread::thread_count();

                Log(Info, "Starting pixel value update from this pass's photons");
                // Update pixel values from this pass's photons
                dr::parallel_for(
                    dr::blocked_range<size_t>(0, nPixels, n_threads_),
                    [&](const dr::blocked_range<size_t> &range)
                    {
                        ScopedSetThreadEnvironment set_env(env);

                        for (size_t i = range.begin(); i != range.end(); ++i)
                        {
                            // process up to 'grain_size' image blocks
                            SPPMPixel &pixel = pixels.at(i);

                            if (int m = pixel.M.load(std::memory_order_relaxed); m > 0 && !(pixel.vp.p.x() != pixel.vp.p.x()))
                            {

                                // Compute new photon count and search radius given photons
                                Float gamma = (Float)2 / (Float)3;
                                Float nNew = pixel.n + gamma * m;
                                Float rNew = pixel.radius * dr::sqrt(nNew / (pixel.n + m));

                                // Update $\tau$ for pixel
                                // Spectrum Phi_i = pixel.Phi;
                                Color3f Phi_i(pixel.Phi_i[0], pixel.Phi_i[1], pixel.Phi_i[2]);
                                pixel.tau = (pixel.tau + Phi_i) * dr::sqr(rNew) / dr::sqr(pixel.radius);

                                // // Set remaining pixel values for next photon pass
                                pixel.n = nNew;
                                pixel.radius = rNew;
                                pixel.M = 0;

                                for (int pp = 0; pp < 3; ++pp)
                                    pixel.Phi_i[pp] = (Float)0;
                            }

                            // Reset _VisiblePoint_ in pixel
                            pixel.vp.beta = Spectrum(0.);
                            pixel.vp.bsdf = nullptr;
                        } // for loop end
                    } // lambda end
                ); // parallel_for end

                // TODO: do this conditionally
                // std::vector<ScalarFloat> Ld_data_(nPixels*3);
                uint64_t Np = (uint64_t)(iter + 1) * (uint64_t)photonsPerIteration;

                Log(Info, "Updating the image data");
                // Periodically write SPPM image to disk
                dr::parallel_for(
                    dr::blocked_range<size_t>(0, nPixels, n_threads_),
                    [&](const dr::blocked_range<size_t> &range)
                    {
                        ScopedSetThreadEnvironment set_env(env);

                        for (size_t i = range.begin(); i != range.end(); ++i)
                        {
                            // process up to 'grain_size' image blocks
                            SPPMPixel &pixel = pixels.at(i);

                            Color3f L = pixel.Ld / (iter + 1) + pixel.tau / (Np * dr::Pi<Float> * dr::sqr(pixel.radius));

                            Ld_data_[i * 3 + 0] = L.x();
                            Ld_data_[i * 3 + 1] = L.y();
                            Ld_data_[i * 3 + 2] = L.z();
                        }
                    });

            } // iterations

            size_t shape_[3] = {film_size.y(), film_size.x(), 3};
            auto data_ = dr::load<DynamicBuffer<ScalarFloat>>(Ld_data_.data(), nPixels * 3);
            result = TensorXf(data_, 3, shape_);
            // put in the film
            ref<ImageBlock> block = film->create_block(film_size, false, false);
            int num_channels = block->channel_count();
            const bool has_alpha = has_flag(film->flags(), FilmFlags::Alpha);

            for (int v = 0; v < film_size.y(); ++v)
            {
                for (int u = 0; u < film_size.x(); ++u)
                {
                    Float *aovs = new Float[num_channels];
                    const int idx = v * film_size.x() + u;
                    aovs[0] = Ld_data_.at(3 * idx + 0);
                    aovs[1] = Ld_data_.at(3 * idx + 1);
                    aovs[2] = Ld_data_.at(3 * idx + 2);

                    if (has_alpha)
                    {
                        aovs[3] = 1.0;
                        aovs[4] = 1.0;
                    }
                    else
                    {
                        aovs[3] = 1.0;
                    }

                    block->put(Point2f(u, v), aovs);
                }
            }
            film->put_block(block);
        }
        else
        {
            Throw("SPPMIntegrator is not supported in JIT mode!");
        }

        if (!m_stop && (evaluate || !dr::is_jit_v<Float>))
            Log(Info, "Rendering finished. (took %s)",
                util::time_string((float)m_render_timer.value(), true));

        return result;
    }

    void
    render_from_camera(const Scene *scene, const Sensor *sensor, Sampler *sampler,
                       std::vector<SPPMPixel> &block, const ScalarPoint2i film_size,
                       const Vector2i spiral_block_size, const Vector2u spiral_block_offset,
                       uint32_t seed, uint32_t block_id, uint32_t block_size, uint32_t iter, uint32_t total_iters)
    {

        // only implement for cpu type
        if constexpr (!dr::is_array_v<Float>)
        {

            // Log(Debug, "--- Rendering block %u/%u (offset: %s, size: %s)",
            //     block_id + 1, block.size(), spiral_block_offset, spiral_block_size);

            // Render a single block of the image
            uint32_t pixel_count = block_size * block_size;
            // Avoid overlaps in RNG seeding RNG when a seed is manually specified
            seed += block_id * pixel_count;

            // Scale down ray differentials when tracing multiple rays per pixel
            // uint32_t sample_count = sampler->sample_count();
            Float diff_scale_factor = dr::rsqrt((Float)total_iters);

            // Determine output channels and prepare the film with this information
            // const Film *film = sensor->film();

            // TODO: remove hardcoded
            // size_t n_channels = film->prepare(aov_names());
            size_t n_channels = 3;
            std::unique_ptr<Float[]> aovs(new Float[n_channels]);
            // Clear block (it's being reused)
            // block->clear();

            for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i)
            {
                // get to the state of a pixel
                sampler->seed(seed + i);

                // get to the state of a sample
                for (uint32_t j = 0; j < iter; ++j)
                {
                    sampler->advance();
                }

                Point2u pos = dr::morton_decode<Point2u>(i);
                if (dr::any(pos >= spiral_block_size))
                    continue;

                Point2f pos_f = Point2f(Point2i(pos) + spiral_block_offset);
                // render a sample
                render_sample(scene, sensor, sampler,
                              block, film_size,
                              aovs.get(),
                              pos_f,
                              diff_scale_factor, true);

                // following is not useful as I am not iterating over samples
                // sampler->advance();
            }
        }
        else
        {
            Throw("SPPMIntegrator::render_block is not supported in JIT mode!");
        }
    }

    void render_sample(const Scene *scene,
                       const Sensor *sensor,
                       Sampler *sampler,
                       std::vector<SPPMPixel> &block,
                       const ScalarPoint2i film_size,
                       Float * /*aovs*/,
                       const Vector2f pos, // global coordinate into array2d
                       Float diff_scale_factor,
                       Mask active) const
    {

        // only implement for cpu type
        if constexpr (!dr::is_array_v<Float>)
        {

            // Log(Info, "Rendering sample at position %s", pos);

            if (pos.x() == 256 && pos.y() == 256)
                Log(Info, "=> BOOM!");

            const Film *film = sensor->film();
            // const bool has_alpha = has_flag(film->flags(), FilmFlags::Alpha);
            // const bool box_filter = film->rfilter()->is_box_filter();

            ScalarVector2f scale = 1.f / ScalarVector2f(film->crop_size()),
                           offset = -ScalarVector2f(film->crop_offset()) * scale;

            Vector2f sample_pos = pos + sampler->next_2d(active),
                     adjusted_pos = dr::fmadd(sample_pos, scale, offset);

            Point2f aperture_sample(.5f);
            if (sensor->needs_aperture_sample())
                aperture_sample = sampler->next_2d(active);

            Float time = sensor->shutter_open();
            if (sensor->shutter_open_time() > 0.f)
                time += sampler->next_1d(active) * sensor->shutter_open_time();

            Float wavelength_sample = 0.f;
            if constexpr (is_spectral_v<Spectrum>)
                wavelength_sample = sampler->next_1d(active);

            auto [ray, ray_weight] = sensor->sample_ray_differential(
                time, wavelength_sample, adjusted_pos, aperture_sample);

            if (ray.has_differentials)
                ray.scale_differential(diff_scale_factor);

            // perform sample step
            // auto [spec, valid] = sample(scene, sampler, ray, medium,
            //    aovs + (has_alpha ? 5 : 4) /* skip R,G,B,[A],W */, active);
            // ----
            // assign SPPMPixel inside sample to reduce memory allocation
            auto [spec, valid] = sample(scene, sampler, ray,
                                        nullptr, // medium
                                        nullptr, // aovs
                                        block, film_size, pos,
                                        active);

            Point2u p = Point2u(dr::floor2int<Point2i>(pos));
            UInt32 index_flattened = dr::fmadd(p.y(), film_size.x(), p.x());
            block[index_flattened].Ld *= ray_weight;
        }
        else
        {
            Throw("SPPMIntegrator::render_sample is not supported in JIT mode!");
        }
    }

    // very similar to path::sample
    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray_, const MediumPtr & /*medium*/,
                                     Float * /*aovs*/,
                                     std::vector<SPPMPixel> &block,
                                     const ScalarPoint2i film_size,
                                     const Vector2f pos, // global coordinate into array2d
                                     Mask active) const
    {

        // only implement for cpu type
        if constexpr (!dr::is_array_v<Float>)
        {

            // Log(Debug, "Sampling ray at position %s", pos);

            // final quantity assignment
            Point2u pos_u = Point2u(dr::floor2int<Point2i>(pos));
            // scatter values into block
            UInt32 index_flattened = dr::fmadd(pos_u.y(), film_size.x(), pos_u.x());

            SPPMPixel &pixel = block.at(index_flattened);

            if (unlikely(m_max_depth == 0))
                return {0.f, false};

            // --------------------- Configure loop state ----------------------

            Ray3f ray = Ray3f(ray_);
            Spectrum throughput = 1.f;
            Spectrum result = 0.f;
            Float eta = 1.f;
            UInt32 depth = 0;
            Bool notSetVisiblePoint = true;

            Mask valid_ray = dr::neq(scene->environment(), nullptr);

            // Variables caching information from the previous bounce
            Interaction3f prev_si = dr::zeros<Interaction3f>();
            Float prev_bsdf_pdf = 1.f;
            Bool prev_bsdf_delta = true;
            BSDFContext bsdf_ctx;

            dr::Loop<Bool> loop("Path Tracer", sampler, ray, throughput, result,
                                eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                                prev_bsdf_delta, active);

            /* Inform the loop about the maximum number of loop iterations.
                This accelerates wavefront-style rendering by avoiding costly
                synchronization points that check the 'active' flag. */
            loop.set_max_iterations(m_max_depth);

            // Log(Debug, "Starting path tracing loop");

            while (loop(active))
            {
                /* dr::Loop implicitly masks all code in the loop using the 'active'
                   flag, so there is no need to pass it to every function */

                SurfaceInteraction3f si =
                    scene->ray_intersect(ray,
                                         /* ray_flags = */ +RayFlags::All,
                                         /* coherent = */ dr::eq(depth, 0u));

                // Log(Debug, "Obtained surface interaction");
                // ---------------------- Direct emission ----------------------

                /* dr::any_or() checks for active entries in the provided boolean
                   array. JIT/Megakernel modes can't do this test efficiently as
                   each Monte Carlo sample runs independently. In this case,
                   dr::any_or<..>() returns the template argument (true) which means
                   that the 'if' statement is always conservatively taken. */
                if (dr::any_or<true>(dr::neq(si.emitter(scene), nullptr)))
                {
                    DirectionSample3f ds(scene, si, prev_si);
                    Float em_pdf = 0.f;

                    if (dr::any_or<true>(!prev_bsdf_delta))
                        em_pdf = scene->pdf_emitter_direction(prev_si, ds,
                                                              !prev_bsdf_delta);

                    // Compute MIS weight for emitter sample from previous bounce
                    Float mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf);

                    // Accumulate, being careful with polarization (see spec_fma)
                    result = spec_fma(
                        throughput,
                        ds.emitter->eval(si, prev_bsdf_pdf > 0.f) * mis_bsdf,
                        result);

                    // add direct emissive surface hit to pixel.Ld
                    pixel.Ld += throughput * mis_bsdf * ds.emitter->eval(si, prev_bsdf_pdf > 0.f);
                }

                // Log(Debug, "Obtained direct emission %s", result);

                // check if si is valid
                if (dr::none_or<false>(si.is_valid()))
                {
                    Log(Debug, "Early exit for invalid surface interaction");
                    break; // early exit for scalar mode
                }

                BSDFPtr bsdf = si.bsdf(ray);

                // Log(Debug, "Obtaining BSDF flags");

                Bool IsDiffuse = has_flag(bsdf->flags(), BSDFFlags::Diffuse);
                Bool IsGlossy = has_flag(bsdf->flags(), BSDFFlags::Glossy);
                Mask inactive = IsDiffuse || (IsGlossy && depth == m_max_depth);

                // Continue tracing the path at this point?
                Bool active_next = (depth + 1 < m_max_depth) && notSetVisiblePoint;

                if (dr::none_or<false>(active_next))
                    break; // early exit for scalar mode

                // Emitter sampling
                // Perform emitter sampling?
                Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

                DirectionSample3f ds = dr::zeros<DirectionSample3f>();
                Spectrum em_weight = dr::zeros<Spectrum>();
                Vector3f wo = dr::zeros<Vector3f>();

                if (dr::any_or<true>(active_em))
                {
                    // Sample the emitter
                    std::tie(ds, em_weight) = scene->sample_emitter_direction(
                        si, sampler->next_2d(), true, active_em);
                    active_em &= dr::neq(ds.pdf, 0.f);

                    /* Given the detached emitter sample, recompute its contribution
                       with AD to enable light source optimization. */
                    if (dr::grad_enabled(si.p))
                    {
                        ds.d = dr::normalize(ds.p - si.p);
                        Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                        em_weight = dr::select(dr::neq(ds.pdf, 0), em_val / ds.pdf, 0);
                    }

                    wo = si.to_local(ds.d);
                }

                // sample direction using BSDF
                Float sample_1 = sampler->next_1d();
                Point2f sample_2 = sampler->next_2d();

                auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight] = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);

                if (dr::any_or<true>(active_em))
                {
                    bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                    // Compute the MIS weight
                    Float mis_em =
                        dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                    // Accumulate, being careful with polarization (see spec_fma)
                    result[active_em] = spec_fma(
                        throughput, bsdf_val * em_weight * mis_em, result);

                    pixel.Ld += throughput * bsdf_val * em_weight * mis_em;
                }

                if (inactive)
                {
                    Point3f curr_p = si.p;
                    float curr_px = curr_p.x(),
                          curr_py = curr_p.y(),
                          curr_pz = curr_p.z();

                    if (curr_px != curr_px || curr_py != curr_py || curr_pz != curr_pz)
                    {
                        Log(Info, "--- problem invalid point: %s", si.p);
                        break;
                    }
                    // pixel.vp = VisiblePoint<Float, Spectrum>(si, si.p, -ray.d, bsdf, throughput, false);
                    pixel.vp = {si, si.p, -ray.d, bsdf, throughput, false};
                }
                // loop over values to change std vector

                notSetVisiblePoint = dr::select(inactive, false, true);

                valid_ray |= active && si.is_valid() &&
                             !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

                if (dr::none_or<true>(notSetVisiblePoint))
                {
                    // Log(Debug, "Visible point set Ld: %s, valid_ray: %s", result, valid_ray);
                    break;
                }

                // BSDF sampling
                bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

                ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

                // ------ Update loop variables based on current interaction ------

                throughput *= bsdf_weight;
                eta *= bsdf_sample.eta;
                // valid_ray |= active && si.is_valid() &&
                //              !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

                // Information about the current vertex needed by the next iteration
                prev_si = si;
                prev_bsdf_pdf = bsdf_sample.pdf;
                prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

                dr::masked(depth, si.is_valid()) += 1;
                Float throughput_max = dr::max(unpolarized_spectrum(throughput));

                active = active_next && dr::neq(throughput_max, 0.f);
            }

            // assign
            // dr::scatter(block.Ld, rgb, index_flattened, active);

            // Log(Debug, "Done sampling ray at position %s - %s", pos, result);
            return {
                /* spec  = */ dr::select(valid_ray, result, 0.f),
                /* valid = */ valid_ray};
        }
        else
        {
            Throw("SPPMIntegrator::render_sample is not supported in JIT mode!");
        }
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const
    {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const
    {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    std::string to_string() const override
    {
        return tfm::format("SPPMIntegrator[\n"
                           "]");
    }

    MI_DECLARE_CLASS()
private:
    /// Size of (square) image blocks to render in parallel (in scalar mode)
    uint32_t m_block_size;
    uint32_t m_max_depth,
        m_rr_depth;
    ScalarFloat m_initial_radius;
};

MI_IMPLEMENT_CLASS_VARIANT(SPPMIntegrator, Integrator)
MI_EXPORT_PLUGIN(SPPMIntegrator, "SPPM integrator");
NAMESPACE_END(mitsuba)