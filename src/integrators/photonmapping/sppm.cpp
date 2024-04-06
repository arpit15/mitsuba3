#include <mutex>

#include <drjit/morton.h>
#include <drjit/dynamic.h>

#include <mitsuba/core/atomic.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/render/kdtree.h>
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

// template <typename T, typename... Args>
// inline void hashRecursiveCopy(char *buf, T v, Args... args) {
//     memcpy(buf, &v, sizeof(T));
//     hashRecursiveCopy(buf + sizeof(T), args...);
// }

// template <typename... Args>
// inline uint64_t Hash(Args... args) {
//     // C++, you never cease to amaze: https://stackoverflow.com/a/57246704
//     constexpr size_t sz = (sizeof(Args) + ... + 0);
//     constexpr size_t n = (sz + 7) / 8;
//     uint64_t buf[n];
//     hashRecursiveCopy((char *)buf, args...);
//     return MurmurHash64A((const unsigned char *)buf, sz, 0);
// }

template <typename Float>
inline uint64_t Hash(const Point<dr::int32_array_t<Float>, 3> &p) {
    return ((p.x() * 73856093) ^ (p.y() * 19349663) ^ (p.z() * 83492791)); 
}

// SPPMPixel
template <typename Float, typename Spectrum>
struct VisiblePoint {

    MI_IMPORT_TYPES(BSDFPtr)

    Point3f p;
    Vector3f wo;
    BSDFPtr bsdf;
    Spectrum beta;
    Bool secondaryLambdaTerminated;
    VisiblePoint(const Point3f &p, const Vector3f &wo, const BSDFPtr &bsdf, const Spectrum &beta, bool secondaryLambdaTerminated)
        : p(p), wo(wo), bsdf(bsdf), beta(beta), secondaryLambdaTerminated(secondaryLambdaTerminated) { }

    DRJIT_STRUCT(VisiblePoint, p, wo, bsdf, beta, secondaryLambdaTerminated)
};

template <typename Float, typename Spectrum>
struct SPPMPixel {

    MI_IMPORT_TYPES()
    using VisiblePoint = VisiblePoint<Float, Spectrum>;

    Float radius = 0.f;
    Color3f Ld;

    VisiblePoint vp;
    
    // TODO: how to make this atomic
    // std::atomic<Spectrum> Phi;
    AtomicFloat<Float> PhiX, PhiY, PhiZ;
    // TODO: how to make this atomic
    std::atomic<UInt32> M;
    Color3f tau;
    Float n = 0;

    DRJIT_STRUCT(SPPMPixel, radius, Ld, vp, PhiX, PhiY, PhiZ, M, tau, n)

};

template <typename Float, typename Spectrum>
struct SPPMPixelListNode {
    SPPMPixel<Float, Spectrum> *pixel;
    SPPMPixelListNode *next;
    SPPMPixelListNode() : pixel(nullptr), next(nullptr) { }
};


template <typename Float, typename Spectrum>
class SPPMIntegrator : public Integrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Integrator, aov_names, should_stop, m_render_timer, m_stop)
    MI_IMPORT_TYPES(Scene, Sensor, ImageBlock, Sampler, Film, MediumPtr, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    using SPPMPixel = SPPMPixel<Float, Spectrum>;
    using BoundingBox2u = BoundingBox<Point2u>;
    using SPPMPixelListNode = SPPMPixelListNode<Float, Spectrum>;
    // using Hash = Hash<Float>;
    // using DynamicArray = dr::DynamicArray<Float>;
    
    SPPMIntegrator(const Properties &props) : Base(props) {
        m_block_size = props.get<uint32_t>("block_size", 0);
        m_max_depth = props.get<uint32_t>("max_depth", 5);
        m_rr_depth = props.get<uint32_t>("rr_depth", 5);
        m_initial_radius = props.get<float>("initial_radius", 1.f);
    }

    std::pair<Ray3f, Spectrum> prepare_ray(const Scene *scene,
                                           const Sensor *sensor,
                                           Sampler *sampler) const {
        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0)
            time += sampler->next_1d() * sensor->shutter_open_time();

        // Prepare random samples.
        Float wavelength_sample  = sampler->next_1d();
        Point2f direction_sample = sampler->next_2d(),
                position_sample  = sampler->next_2d();

        // Sample one ray from an emitter in the scene.
        auto [ray, ray_weight, emitter] = scene->sample_emitter_ray(
            time, wavelength_sample, direction_sample, position_sample);

        return { ray, ray_weight };
    }

    TensorXf render(Scene *scene,
                    Sensor *sensor,
                    uint32_t seed = 0,
                    uint32_t spp = 0,
                    bool develop = true,
                    bool evaluate = true) override {
        
        // Final result output                        
        TensorXf result;

        Log(Info, "scene bounds: %s", scene->bbox());

        // implement only for cpu type
if constexpr (!dr::is_jit_v<Float>) {
        // Render on the CPU using a spiral pattern

        // Potentially adjust the number of samples per pixel if spp != 0
        Sampler *sampler = sensor->sampler();
        if (spp)
            sampler->set_sample_count(spp);
        spp = sampler->sample_count();
        
        Log(Info, "spp: %u", spp);
        uint32_t num_iter = spp;
        // const Float invSqrtSPP = Float(1.0) / std::sqrt(spp);

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
        uint32_t n_threads = (uint32_t) Thread::thread_count();
        // If no block size was specified, find size that is good for parallelization
        uint32_t block_size = m_block_size;
        if (block_size == 0) {
            block_size = MI_BLOCK_SIZE; // 32x32
            while (true) {
                // Ensure that there is a block for every thread
                if (block_size == 1 || dr::prod((film_size + block_size - 1) /
                                                 block_size) >= n_threads)
                    break;
                block_size /= 2;
            }
        }

        // SPPM pixels
        Log(Info, "Starting visible point render job (%ux%u)",
        film_size.x(), film_size.y());

        // Create a 2D grid of SPPM pixels
        // dr::DynamicArray<SPPMPixel> pixels;
        // dr::set_slices(pixels, nPixels);
        // unclear how to allocate memory for SPPMPixel vector
        std::vector<SPPMPixel> pixels(nPixels);

        // assign radius
        for(uint32_t i=0; i < nPixels; ++i) {
            // pixels[i].radius = m_initial_radius;
            // dr::slice(pixels, i).radius = m_initial_radius;
            pixels[i].radius = m_initial_radius;
        }
        
        
        Spiral spiral(film_size, film->crop_offset(), block_size, 1);

        std::mutex mutex;
        ref<ProgressReporter> progress;
        Logger* logger = mitsuba::Thread::thread()->logger();
        if (logger && Info >= logger->log_level())
            progress = new ProgressReporter("Stage1: Visible Points");

        // Total number of blocks to be handled, including multiple passes.
        uint32_t total_blocks = spiral.block_count() * 1,
                 blocks_done = 0;

        // Grain size for parallelization
        uint32_t grain_size = std::max(total_blocks / (4 * n_threads), 1u);

    Log(Info, "About to start the main loop for %u", num_iter);
    std::vector<ScalarFloat> Ld_data_(nPixels*3);
    // run main loop
    for(uint32_t iter=0; iter < num_iter; ++iter) {
        Log(Info, "Starting iteration %u/%u", iter + 1, num_iter);
        // Generate SPPM visible points
        ThreadEnvironment env;
        dr::parallel_for(
            dr::blocked_range<uint32_t>(0, total_blocks, grain_size),
            [&](const dr::blocked_range<uint32_t> &range) {
                ScopedSetThreadEnvironment set_env(env);

                // Fork a non-overlapping sampler for the current worker
                ref<Sampler> sampler = sensor->sampler()->fork();

                // Render up to 'grain_size' image blocks
                for (uint32_t i = range.begin();
                    i != range.end() &&!should_stop(); ++i) {

                        auto [offset, size, block_id] = spiral.next_block();
                        Assert(dr::prod(size) != 0);

                        Log(Debug, "Rendering block %u/%u (offset: %s, size: %s)",
                            i + 1, total_blocks, offset, size);

                        // TODO: check if the following should be uncommented
                        // if (film->sample_border())
                        //     offset -= film->rfilter()->border_size();

                        // block->set_size(size);
                        // block->set_offset(offset);

                        Vector2i spiral_block_size = size;
                        Vector2u spiral_block_offset = offset;

                        // main path tracing style step
                        render_from_camera(scene, sensor, sampler, pixels, 
                            film_size, spiral_block_size, spiral_block_offset, 
                            seed, block_id, block_size);
                        // store result into global storage

                        //
                        
                        /* Critical section: update progress bar */
                        if(progress) {
                            std::lock_guard<std::mutex> lock(mutex);
                            blocks_done++;
                            progress->update(blocks_done / (Float) total_blocks);
                        }


                    } // for loop end
            
            } // lambda end
        ); // parallel_for end
            
        // DEBUG: output Ld
        // size_t shape[3] = { film_size.y(), film_size.x(), 3};
        // // TODO: convert to dynamicArray
        // // result = TensorXf(pixels.Ld , 3, shape);
        // std::vector<ScalarFloat> Ld_data(nPixels*3);
        // for(uint32_t i=0; i < nPixels; ++i) {
        //     Ld_data[i*3] = pixels[i].Ld.x();
        //     Ld_data[i*3+1] = pixels[i].Ld.y();
        //     Ld_data[i*3+2] = pixels[i].Ld.z();
        // }
        // auto data = dr::load<DynamicBuffer<ScalarFloat>>(Ld_data.data(), nPixels*3);
        // result = TensorXf(data, 3, shape);
        // return result;
        // ----- end of debug

        
        // Create grid of all SPPM visible points
        // Allocate grid for SPPM visible points
        // uint32_t hashSize = math::next_prime(nPixels);
        uint32_t hashSize = nPixels;
        std::vector<std::atomic<SPPMPixelListNode *>> grid(hashSize);
        std::vector<std::atomic<int> > gridCounter(hashSize);

        // Compute grid bounds for SPPM visible points
        BoundingBox3f gridBounds;
        Float maxRadius = 0.f;
        for (const auto &pixel : pixels) {
            if (! dr::mean(pixel.vp.beta))
                continue;

            auto vpBounds = BoundingBox3f(
                pixel.vp.p - Vector3f(pixel.radius),
                pixel.vp.p + Vector3f(pixel.radius)
            );

            Log(Debug, "Visible point p: %s, beta: %s", pixel.vp.p, pixel.vp.beta);

            gridBounds.expand(vpBounds);
            maxRadius = dr::maximum(maxRadius, pixel.radius);
        }

        

        // std::unique_ptr<detail::OrderedChunkAllocator> threadScratchBuffer;
        detail::OrderedChunkAllocator threadScratchBuffer;

        // Compute resolution of SPPM grid in each dimension
        // UInt32 gridRes[3];
        Vector3f diag = gridBounds.extents();
        Float maxDiag = dr::max(diag);
        int baseGridRes = int(maxDiag / maxRadius);
        Point3i gridRes = dr::maximum(1,  dr::floor2int<Vector3f>(baseGridRes * diag / maxDiag));

        Log(Info, "Hashing %u visible points into a %s, gridBounds: %s",
            nPixels, gridRes, gridBounds
        );

        // Add visible points to SPPM grid
        // iterate over pixels
        // run over rows first
        dr::parallel_for(
            dr::blocked_range<uint32_t>(0, nPixels, 8),
            [&](const dr::blocked_range<uint32_t> &range){
                ScopedSetThreadEnvironment set_env(env);
                // process up to 'grain_size' image blocks

                // obtain pointer to scratch buffer
                auto &scratchBuffer = threadScratchBuffer;
                // ----

                for (uint32_t i = range.begin();
                    i != range.end() &&!should_stop(); ++i) {
                        
                    uint32_t px_id = i;
                    Point2u block_px_start(px_id % film_size.x(), px_id / film_size.x());

                    SPPMPixel &pixel = pixels[px_id];
                    if ( dr::mean(pixel.vp.beta)) {
                        // Add pixel's visible point to applicable grid cells
                        // Find grid cell bounds for pixel's visible point, _pMin_ and _pMax_
                        Float pixelRadius = pixel.radius;
                        // Point3i pMin, pMax;
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
                        for (int z = pMin.z(); z <= pMax.z(); ++z) { 
                            for (int y = pMin.y(); y <= pMax.y(); ++y) { 
                                for (int x = pMin.x(); x <= pMax.x(); ++x) {
                                    // Add visible point to grid cell $(x, y, z)$
                                    int h = Hash<Float>(Point3i(x, y, z)) % hashSize;
                                    // allocate nodes using global memory so that they don't vanish when the loop ends
                                    SPPMPixelListNode *node = scratchBuffer.allocate<SPPMPixelListNode>(1);
                                    // linked list
                                    node->pixel = &pixel;

                                    // change the grid node counter
                                    gridCounter[h]++;
                                    // add an entry to linked list in the SPPMNode
                                    node->next = grid[h];
                                    while (!grid[h].compare_exchange_weak(node->next, node))
                                        ;
                                }
                            }
                        }
                        

                    }
            
                }
            }
        );

        if (logger && Info >= logger->log_level())
            progress = new ProgressReporter("Stage3: Photon Tracing");

        uint32_t totalPhotonsBlocks = nPixels/8;
        uint32_t photonBlocksDone = 0;

        int photonsPerIteration = nPixels;
        Log(Info, "Starting photon tracing pass, with %u photons", photonsPerIteration);
        // Trace photons and accumulate contributions
        dr::parallel_for(
            dr::blocked_range<size_t>(0, photonsPerIteration, 8),
            [&](const dr::blocked_range<size_t> &range) {
                ScopedSetThreadEnvironment set_env(env);

                // auto &scratchBuffer = photonShootScratchBuffers;
                // Fork a non-overlapping sampler for the current worker
                ref<Sampler> sampler = sensor->sampler()->clone();

                Log(Debug, "Photon tracing from %u to %u", range.begin(), range.end());

                // process up to 'grain_size' image blocks
                for (size_t photonIndex = range.begin();
                    photonIndex != range.end() &&!should_stop(); ++photonIndex) {

                    Log(Debug, "-- Tracing photon %u", photonIndex);

                    // generate photon rays
                    auto [ray, throughput] = prepare_ray(scene, sensor, sampler);

                    if(dr::all(throughput == 0.f))
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
                    
                    while (loop(active)) {

                        Log(Debug, "+++++ Tracing photon %u depth %u, p: %s", 
                            photonIndex, depth, si.p);
                        // contribute to visible points
                        Point3i photonGridIndex = Point3i(
                            dr::floor2int<Point3i>(gridRes * (si.p - gridBounds.min) / gridBounds.extents())
                        );

                        // check inbounds
                        if (dr::all(photonGridIndex >= 0 && photonGridIndex < gridRes)) {
                            // Compute the hash value for the grid cell
                            int h = Hash<Float>(photonGridIndex) % hashSize;

                            Log(Debug, "+++++ Photon %u depth %u, grid index: %s, hash: %u, numPhotonsInCell: %u",
                                photonIndex, depth, photonGridIndex, h, gridCounter[h].load());

                            const int PhotonsInList = gridCounter[h].load();

                            int photon_id = 0;
                            for(SPPMPixelListNode *node = 
                                    grid[h].load(std::memory_order_relaxed);
                                    node; node = node->next) {
                                
                                Log(Debug, "+++++ Photon %u depth %u, grid index: %s, hash: %u, photon_id: %u/%u",
                                    photonIndex, depth, photonGridIndex, h, photon_id, PhotonsInList);
                                SPPMPixel &pixel = *node->pixel;
                                // not a neighbor
                                if (dr::squared_norm(pixel.vp.p - si.p) > dr::sqr(pixel.radius))
                                    continue;

                                // Update pixel Phi and M for nearby photon
                                Vector3f wi = -ray.d;
                                BSDFContext ctx;
                                Point2f _uv(0.0, 0.0);
                                // TODO: this seems wrong
                                Normal3f _n = Normal3f(0.0, 0.0, 0.0);
                                PositionSample3f _ps(pixel.vp.p, _n, _uv, ray.time, 0.0, false);  
                                SurfaceInteraction3f pixel_si(_ps, ray.wavelengths);
                                pixel_si.p = pixel.vp.p;
                                pixel_si.wi = pixel.vp.wo;   
                                Spectrum Phi = throughput * pixel.vp.bsdf->eval(ctx, pixel_si, wi);

                                // TODO: account for sampled wavelengths

                                Spectrum Phi_i = pixel.vp.beta * Phi;
                                Float Phi_i_X = Phi_i.x();
                                Float Phi_i_Y = Phi_i.y();
                                Float Phi_i_Z = Phi_i.z();
                                // atomic add
                                // pixel.PhiX += Phi_i_X;
                                pixel.PhiX += (Phi_i_X);
                                pixel.PhiY += (Phi_i_Y);
                                pixel.PhiZ += (Phi_i_Z);
                            
                                photon_id++;
                            } // iteration over linked list end

                        } // inbounds check
                            
                    
                        // continue tracing or decide to close some off
                        /* ----------------------- BSDF sampling ------------------------ */
                        // Sample BSDF * cos(theta).
                        BSDFPtr bsdf = si.bsdf(ray);
                        BSDFContext ctx(TransportMode::Importance);
                        auto [bs, bsdf_val] =
                            bsdf->sample(ctx, si, sampler->next_1d(active),
                                        sampler->next_2d(active), active);

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
                        if (dr::any_or<true>(use_rr)) {
                            Float q = dr::minimum(
                                dr::max(unpolarized_spectrum(throughput)) * dr::sqr(eta), 0.95f);
                            dr::masked(active, use_rr) &= sampler->next_1d(active) < q;
                            dr::masked(throughput, use_rr) *= dr::rcp(q);
                        }

                    } // tracing loop
                    

                    // reset scratch space
                    // TODO: how to provide the specific pointer without keeping the list 
                    // scratchBuffer.release();

                } // while loop end

                /* Critical section: update progress bar */
                if(progress) {
                    std::lock_guard<std::mutex> lock(mutex);
                    photonBlocksDone++;
                    progress->update(photonBlocksDone / (Float) totalPhotonsBlocks);
                }

            }
        );
        
        // reset scratch space after tracing photons


        // update counters
        // photonPaths += photonsPerIteration;

        Log(Info, "Starting image update pass");
        // Update pixel values from this pass's photons
        dr::parallel_for(
            dr::blocked_range<size_t>(0, nPixels, 8),
            [&](const dr::blocked_range<size_t> &range) {
                ScopedSetThreadEnvironment set_env(env);

                for(size_t i = range.begin(); i != range.end(); ++i) {
                    // process up to 'grain_size' image blocks
                    SPPMPixel &pixel = pixels.at(i);
                    if (int m = pixel.M.load(std::memory_order_relaxed); m>0) {

                        // Compute new photon count and search radius given photons
                        Float gamma = (Float)2 / (Float)3;
                        Float nNew = pixel.n + gamma * m;
                        Float rNew = pixel.radius * std::sqrt(nNew / (pixel.n + m));

                        // Update $\tau$ for pixel
                        // Spectrum Phi_i = pixel.Phi;
                        Color3f Phi_i(pixel.PhiX, pixel.PhiY, pixel.PhiZ);
                        pixel.tau = (pixel.tau + Phi_i) * dr::sqrt(rNew) / dr::sqrt(pixel.radius);

                        // // Set remaining pixel values for next photon pass
                        pixel.n = nNew;
                        pixel.radius = rNew;
                        pixel.M = 0;
                    }

                    // Reset _VisiblePoint_ in pixel
                    pixel.vp.beta = Spectrum(0.);
                    pixel.vp.bsdf = nullptr;
                } // for loop end
            } // lambda end
        ); // parallel_for end

        // TODO: do this conditionally
        // std::vector<ScalarFloat> Ld_data_(nPixels*3);
        uint64_t np = (uint64_t)(iter + 1) * (uint64_t)photonsPerIteration;
        
        Log(Info, "Updating the writing of images");
        // Periodically write SPPM image to disk
        dr::parallel_for(
            dr::blocked_range<size_t>(0, nPixels, 8),
            [&](const dr::blocked_range<size_t> &range) {
                ScopedSetThreadEnvironment set_env(env);

                for(size_t i = range.begin(); i != range.end(); ++i) {
                    // process up to 'grain_size' image blocks
                    SPPMPixel &pixel = pixels[i];

                    Color3f L = pixel.Ld / (iter + 1) + pixel.tau / (np * dr::Pi<Float> * dr::sqrt(pixel.radius));
                    
                    Ld_data_[i*3+0] = L.x();
                    Ld_data_[i*3+1] = L.y();
                    Ld_data_[i*3+2] = L.z();
                    
                }
            }
        );


        } // iterations

        size_t shape_[3] = { film_size.y(), film_size.x(), 3};
        // result = TensorXf(pixels.Ld , 3, shape);
        // std::vector<ScalarFloat> Ld_data(nPixels*3);
        // for(uint32_t i=0; i < nPixels; ++i) {
        //     Ld_data[i*3] = pixels[i].Ld.x();
        //     Ld_data[i*3+1] = pixels[i].Ld.y();
        //     Ld_data[i*3+2] = pixels[i].Ld.z();
        // }
        auto data_ = dr::load<DynamicBuffer<ScalarFloat>>(Ld_data_.data(), nPixels*3);
        result = TensorXf(data_, 3, shape_);
        // put in the film
        ref<ImageBlock> block = film->create_block(film_size, false, false);
        Log(Info, "film block numCh: %s, size: %s", 
                block->channel_count(),
                block->size());
        int num_channels = block->channel_count();
        const bool has_alpha = has_flag(film->flags(), FilmFlags::Alpha);

        for (int v=0; v<film_size.y(); ++v) {
            for (int u=0; u<film_size.x(); ++u) {
                Float* aovs = new Float[num_channels]; 
                const int idx = v*film_size.x() + u;
                aovs[0] = Ld_data_.at( 3*idx + 0);
                aovs[1] = Ld_data_.at( 3*idx + 1);
                aovs[2] = Ld_data_.at( 3*idx + 2);
                // Log(Debug, "(%d, %d) -> (%f, %f, %f)", u, v, aovs[0], aovs[1], aovs[2]);
                if (has_alpha){
                    aovs[3] = 1.0;
                    aovs[4] = 1.0;
                } else {
                    aovs[3] = 1.0;
                }

                block->put(
                    // Point2f((Float)u/(Float)film_size.x(), (Float)v/(Float)film_size.y()), 
                    Point2f(u, v),
                    aovs);
            }
        }
        film->put_block(block);
        // 
        return result;

} else {
        Throw("SPPMIntegrator is not supported in JIT mode!");
}

        if (!m_stop && (evaluate || !dr::is_jit_v<Float>))
            Log(Info, "Rendering finished. (took %s)",
                util::time_string((float) m_render_timer.value(), true));

        return result;

    }
    
    void render_from_camera(const Scene *scene, const Sensor *sensor, Sampler *sampler, 
        std::vector<SPPMPixel> &block, const ScalarPoint2i film_size, 
        const Vector2i spiral_block_size, const Vector2u spiral_block_offset,
        uint32_t seed, uint32_t block_id, uint32_t block_size) {
        
        // only implement for cpu type
if constexpr (!dr::is_array_v<Float>) {

        Log(Debug, "--- Rendering block %u/%u (offset: %s, size: %s)",
            block_id + 1, block.size(), spiral_block_offset, spiral_block_size);
        
        // Render a single block of the image
        uint32_t pixel_count = block_size * block_size;
        // Avoid overlaps in RNG seeding RNG when a seed is manually specified
        seed += block_id * pixel_count;

        // Scale down ray differentials when tracing multiple rays per pixel
        uint32_t sample_count = sampler->sample_count();
        Float diff_scale_factor = dr::rsqrt((Float) sample_count);

        // Determine output channels and prepare the film with this information
        // const Film *film = sensor->film();

        // TODO: remove hardcoded 
        // size_t n_channels = film->prepare(aov_names());
        size_t n_channels = 3;
        std::unique_ptr<Float[]> aovs(new Float[n_channels]);
        // Clear block (it's being reused)
        // block->clear();

        for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
            sampler->seed(seed + i);

            Point2u pos = dr::morton_decode<Point2u>(i);
            if (dr::any(pos.x() >= spiral_block_size || pos.y() >= spiral_block_size))
                continue;

            Point2f pos_f = Point2f(Point2i(pos) + spiral_block_offset);
            // render a sample
            render_sample(scene, sensor, sampler, 
                block, film_size, 
                aovs.get(),
                pos_f, 
                diff_scale_factor, true);

            sampler->advance();
        }

} else {
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
                        Mask active) const {
        
        // only implement for cpu type
if constexpr (!dr::is_array_v<Float>) {

        Log(Debug, "Rendering sample at position %s", pos);

        const Film *film = sensor->film();
        // const bool has_alpha = has_flag(film->flags(), FilmFlags::Alpha);
        // const bool box_filter = film->rfilter()->is_box_filter();

        ScalarVector2f scale = 1.f / ScalarVector2f(film->crop_size()),
                    offset = -ScalarVector2f(film->crop_offset()) * scale;

        Vector2f sample_pos   = pos + sampler->next_2d(active),
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
        // 

        Log(Debug, "Obtained spectrum %s, ray_weight: %s", spec, ray_weight);
        UnpolarizedSpectrum spec_u = unpolarized_spectrum(ray_weight * spec);

        Spectrum rgb = spec_u;
        // if constexpr (is_spectral_v<Spectrum>)
        //     rgb = spectrum_to_srgb(spec_u, ray.wavelengths, active);
        // else if constexpr (is_monochromatic_v<Spectrum>)
        //     rgb = spec_u.x();
        // else
        // rgb = spec_u;

        // aovs[0] = rgb.x();
        // aovs[1] = rgb.y();
        // aovs[2] = rgb.z();
        // aovs[3] = 1.f;

        // put the result
        // With box filter, ignore random offset to prevent numerical instabilities
        // block->put(box_filter ? pos : sample_pos, aovs, active);
        // put operation in dynamicarray

        Point2u p = Point2u(dr::floor2int<Point2i>(pos));
        UInt32 index_flattened = dr::fmadd(p.y(), film_size.x(), p.x());

        Log(Debug, "Assigning Ld at position %s - %s", pos, rgb);
        // TODO: convert this to drjit operation
        block[index_flattened].Ld.x() = rgb.x();
        block[index_flattened].Ld.y() = rgb.y();
        block[index_flattened].Ld.z() = rgb.z();
        // dr::scatter(block.Ld, rgb, index_flattened, active);

} else {
    Throw("SPPMIntegrator::render_sample is not supported in JIT mode!");
}

    }

    // very similar to path::sample
    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler, 
        const RayDifferential3f &ray_, const MediumPtr & /*medium*/, 
        Float * /*aovs*/, 
            std::vector<SPPMPixel> &block, 
            const ScalarPoint2i film_size, 
            const Vector2f pos,  // global coordinate into array2d
        Mask active) const {
        
               // only implement for cpu type
if constexpr (!dr::is_array_v<Float>) {

    Log(Debug, "Sampling ray at position %s", pos);

    // final quantity assignment 
    Point2u pos_u = Point2u(dr::floor2int<Point2i>(pos));
    // scatter values into block
    UInt32 index_flattened = dr::fmadd(pos_u.y(), film_size.x(), pos_u.x());

    if (unlikely(m_max_depth == 0))
        return { 0.f, false };

    // --------------------- Configure loop state ----------------------

    Ray3f ray                     = Ray3f(ray_);
    Spectrum throughput           = 1.f;
    Spectrum result               = 0.f;
    Float eta                     = 1.f;
    UInt32 depth                  = 0;
    Bool notSetVisiblePoint       = true;

    Mask valid_ray = dr::neq(scene->environment(), nullptr);
    
    // Variables caching information from the previous bounce
    Interaction3f prev_si         = dr::zeros<Interaction3f>();
    Float         prev_bsdf_pdf   = 1.f;
    Bool          prev_bsdf_delta = true;
    BSDFContext   bsdf_ctx;

    dr::Loop<Bool> loop("Path Tracer", sampler, ray, throughput, result,
                            eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                            prev_bsdf_delta, active);

    /* Inform the loop about the maximum number of loop iterations.
        This accelerates wavefront-style rendering by avoiding costly
        synchronization points that check the 'active' flag. */
    loop.set_max_iterations(m_max_depth);

    // Log(Debug, "Starting path tracing loop");

    while (loop(active)) {
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
            if (dr::any_or<true>(dr::neq(si.emitter(scene), nullptr))) {
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
            }

            // Log(Debug, "Obtained direct emission %s", result);
            
            // check if si is valid
            if (dr::none_or<false>(si.is_valid())){
                Log(Debug, "Early exit for invalid surface interaction");
                break; // early exit for scalar mode
            }

            BSDFPtr bsdf = si.bsdf(ray);

            // Log(Debug, "Obtaining BSDF flags");

            Bool IsDiffuse = has_flag(bsdf->flags(), BSDFFlags::Diffuse);
            Bool IsGlossy = has_flag(bsdf->flags(), BSDFFlags::Glossy);
            Mask inactive = IsDiffuse || (IsGlossy && depth == m_max_depth);

            
        
            // Continue tracing the path at this point?
            Bool active_next = (depth+1 < m_max_depth) && notSetVisiblePoint;

            if (dr::none_or<false>(active_next))
                break; // early exit for scalar mode

            // Emitter sampling
            // Perform emitter sampling?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, sampler->next_2d(), true, active_em);
                active_em &= dr::neq(ds.pdf, 0.f);

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(dr::neq(ds.pdf, 0), em_val / ds.pdf, 0);
                }

                wo = si.to_local(ds.d);
            }

            // sample direction using BSDF
            Float sample_1 = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight]
                = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);
            
            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate, being careful with polarization (see spec_fma)
                result[active_em] = spec_fma(
                    throughput, bsdf_val * em_weight * mis_em, result);
            }

            // assign visible point here
            Log(Debug, "light sampling result: %s", result);
            // dr::scatter(block.vp, 
            //     VisiblePoint(si.p, -ray.d, bsdf, throughput, false)
            //     , index_flattened, inactive);

            if (inactive) {
                Log(Debug, "assigning visible point at position %s", pos);
                block[index_flattened].vp = VisiblePoint<Float, Spectrum>(si.p, -ray.d, bsdf, throughput, false);
            }
            // loop over values to change std vector

            notSetVisiblePoint = dr::select(inactive, false, true);

            valid_ray |= active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);


            if(dr::none_or<true>(notSetVisiblePoint)){
                Log(Debug, "Visible point set Ld: %s, valid_ray: %s", result, valid_ray);
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
    
    Log(Debug, "Done sampling ray at position %s - %s", pos, result);
    return {
            /* spec  = */ dr::select(valid_ray, result, 0.f),
            /* valid = */ valid_ray
        };

} else {
    Throw("SPPMIntegrator::render_sample is not supported in JIT mode!");
}

    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    std::string to_string() const override {
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