#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/mlt/mlt.h>
#include <mitsuba/render/mlt/sampler.h>
// #include <mitsuba/render/ray_export.h>

#if defined(MTS_PSSMLT_USE_EXPECTED_VALUES) && !MTS_PSSMLT_USE_EXPECTED_VALUES
static_assert("Not supported: MTS_PSSMLT_USE_EXPECTED_VALUES == 0");
#endif

NAMESPACE_BEGIN(mitsuba)

void MLTSampler::ensure_ready(size_t sample_index) {
    // Ensure we have enough sample objects
    if (sample_index >= m_samples.size())
        m_samples.resize(sample_index + 1);
    auto &sample = m_samples[sample_index];

    // Apply any large step that took place since the last update
    if (sample.last_modified < m_last_large_step_iteration || !sample.ready) {
        sample.value         = m_rng.next_float32();
        sample.last_modified = m_last_large_step_iteration;
        sample.ready         = true;
    }

    // Apply all remaining mutations at once
    sample.backup();
    size_t missing_iterations = m_current_iteration - sample.last_modified;
    if (m_is_large_step) {
        sample.value = m_rng.next_float32();
    } else if (missing_iterations > 0) {
        // Start from a standard normally distributed proposal N(0, 1),
        // and scale it to match to `missing_iterations` successive iterations
        sample.value += detail::normal_sample(
            m_rng.next_float32(), m_sigma * sqrt((Float) missing_iterations));
        sample.value -= std::floor(sample.value);
    }
    sample.last_modified = m_current_iteration;
}

PSSMLTIntegrator::PSSMLTIntegrator(const Properties &props)
    : MonteCarloIntegrator(props), m_sub_integrator(nullptr),
      m_direct_pass_image(nullptr), m_lum_map(nullptr),
      m_lum_map_epsilon(1e-2) {

    m_max_depth = props.long_("max_depth", 6);

    m_bootstrap_samples_count = props.size_("bootstrap_samples_count", 100000);
    m_bootstrap_max_depth     = props.size_("bootstrap_max_depth", 5);
    m_use_bootstrap_in_luminance_estimates =
        props.bool_("use_bootstrap_in_luminance_estimates", true);
    m_base_seed   = props.size_("base_seed", 1);
    m_chain_count = props.size_("chain_count", __global_thread_count);
    if (m_chain_count == 0)
        m_chain_count = __global_thread_count > 0 ? __global_thread_count
                                                  : util::core_count();
    m_mutations_per_pixel    = props.size_("mutations_per_pixel", 100);
    m_sigma                  = props.float_("sigma", 0.01f);
    m_large_step_probability = props.float_("large_step_probability", 0.3f);

    m_splat_all_mutations = props.bool_("splat_all_mutations", true);
    if (!m_splat_all_mutations) {
        Log(EInfo, "Will only splat the main mutation for the given algorithm "
                   "(not large steps or offset resampling steps).");
    }

    for (auto &kv : props.objects()) {
        auto *integrator = dynamic_cast<SamplingIntegrator *>(kv.second.get());

        if (integrator) {
            if (m_integrator)
                Throw("Cannot specify more than one integrator.");
            m_integrator = integrator;
        }
    }

    // (Optional) luminance map.
    m_lum_map_path    = props.string("luminance_map", "");
    m_flat_image_path = props.string("flat_image_path", "");
    // Hardcoded average luminance (for debugging).
    m_hardcoded_average_luminance =
        props.float_("hardcoded_average_luminance", 0.f);
    m_use_mlt_normalization = props.bool_("m_use_mlt_normalization", true);

    // D-dimensional hyperplane-based gradients
    m_use_gradients_from_fit = props.bool_("use_gradients_from_fit", false);
    // 1-dimensional weighted least-squares-based gradients
    m_use_new_gradients = props.bool_("use_new_gradients", false);
    if (m_use_new_gradients)
        Log(EWarn, "Will use the new gradient estimator");
    if (m_use_gradients_from_fit && m_use_new_gradients)
        Throw("Must choose one of: use_gradients_from_fit, use_new_gradients");

    m_diminish_gradients_from =
        props.size_("diminish_gradients_from", 10000000);

    // Zero-out gradients in dimensions corresponding to zero-valued terms
    m_use_nonzero_gradients = props.bool_("use_nonzero_gradients", false);
    if (m_use_nonzero_gradients)
        Log(EWarn, "Will zero-out gradients for zero-valued terms and use the "
                   "path_null integrator");

    // Target function: adding background energy
    m_estimated_average_lum = std::numeric_limits<Float>::quiet_NaN();
    m_background_energy = props.float_("background_energy", 0.f);
    if (m_background_energy > 0.f)
        Log(EWarn,
            "Adding background energy to the target function: %f%% of the "
            "estimated average radiance.",
            m_background_energy * 100.f);
    else if (m_background_energy < 0.f)
        Throw("Background energy level cannot be negative: %f",
              m_background_energy);
    // Target function: log transform
    m_tw_use_log_transform = props.bool_("tw_use_log_transform", false);
    if (m_tw_use_log_transform) {
        Log(EInfo, "Using the log transform, offset is %f",
            m_background_energy);
    }
    // Target function: gamma transform
    m_tw_use_gamma_transform = props.bool_("tw_use_gamma_transform", false);
    m_tw_gamma_exponent      = props.float_("tw_gamma_exponent", 0.5f);
    if (m_tw_use_gamma_transform) {
        Log(EInfo, "Using the gamma transform, offset = %f and exponent = %f ",
            m_background_energy, m_tw_gamma_exponent);
    }
    if (m_tw_use_log_transform && m_tw_use_gamma_transform)
        Throw("Can't apply both log and gamma transforms");

    m_use_finite_differences = props.bool_("use_finite_differences", false);
    if (m_use_finite_differences)
        Log(EWarn, "Using finite differences to compute gradients (slow).");

    if (m_use_gradients_from_fit &&
        (m_tw_use_gamma_transform || m_tw_use_log_transform ||
         m_use_finite_differences))
        Throw("Incompatible combination of gradients computation mode and "
              "features (log / gamma / fit gradients)");

    if (!m_integrator) {
        // No integrator specified, create a default one.
        Properties props(m_use_nonzero_gradients ? "path_null" : "path");
        props.set_long("max_depth", m_max_depth);
        m_integrator =
            PluginManager::instance()->create_object<SamplingIntegrator>(props);
        Log(EInfo, "Default underlying integrator created: %s",
            m_integrator->class_()->name());
    }

    m_handle_direct = props.bool_("handle_direct", false);
    if (!m_handle_direct) {
        m_direct_shading_samples = props.size_("direct_shading_samples", 16);
        m_integrator->set_ignore_direct(true);
    } else if (props.has_property("direct_shading_samples")) {
        Throw("Cannot specify `direct_shading_samples` with "
              "`handle_direct` = true");
    }

    m_direct_pass_path = props.string("direct_pass_path", "");
    if (!m_direct_pass_path.empty()) {
        if (m_handle_direct) {
            Log(EWarn,
                "Ignoring provided direct illumination image (%s) since "
                "handle_direct = true",
                m_direct_pass_path);
            m_direct_pass_path = "";
        }
    }
}

void PSSMLTIntegrator::start_progress_reporter(std::string name,
                                               size_t total_work) {
    m_progress      = new ProgressReporter(name);
    m_total_work    = total_work;
    m_work_complete = 0;
    m_progress->update(0.0f);
}
void PSSMLTIntegrator::report_progress(size_t units) {
    tbb::spin_mutex::scoped_lock lock(m_progress_mutex);
    m_work_complete += units;
    m_progress->update(m_work_complete / (Float) m_total_work);
}

ref<ReplayableSampler> PSSMLTIntegrator::make_base_sampler(size_t seed) {
    ref<ReplayableSampler> sampler(
        new MLTSampler(seed, m_sigma, m_large_step_probability));
    return sampler;
}

void PSSMLTIntegrator::load_luminance_map(const Film *film) {
    if (m_lum_map_path.size() <= 0)
        return;

    auto *fresolver = Thread::thread()->file_resolver();
    ref<Bitmap> lum_map =
        new Bitmap(fresolver->resolve(fs::path(m_lum_map_path)));
    if (lum_map->size() != film->size())
        Throw("Invalid luminance map size: expected %s, found %s.",
              film->size(), lum_map->size());

    m_lum_map = lum_map->convert(Bitmap::EY, Struct::EFloat, false);

    Log(EInfo, "Loaded luminance map: %s (%d x %d)", m_lum_map_path,
        m_lum_map->width(), m_lum_map->height());
}

void PSSMLTIntegrator::load_direct_pass_image(const Film *film) {
    if (m_direct_pass_path.empty() || m_direct_pass_path == "none")
        return;

    auto *fresolver = Thread::thread()->file_resolver();
    m_direct_pass_image =
        new Bitmap(fresolver->resolve(fs::path(m_direct_pass_path)));
    if (m_direct_pass_image->size() != film->size())
        Throw("Invalid direct pass image size: expected %s, found %s.",
              film->size(), m_direct_pass_image->size());

    m_direct_pass_image =
        m_direct_pass_image->convert(Bitmap::EXYZAW, Struct::EFloat, false);
    Log(EInfo, "Loaded direct pass image: %s (%d x %d)", m_direct_pass_path,
        m_direct_pass_image->width(), m_direct_pass_image->height());
}

void PSSMLTIntegrator::make_bootstrap_sample(size_t i, RadianceSample3f &rs,
                                             Float *bootstrap_weights) {
    auto *controlled_sampler = dynamic_cast<ReplayableSampler *>(rs.sampler);
    for (size_t depth = 0; depth <= m_bootstrap_max_depth; ++depth) {
        size_t rng_index = i * (m_bootstrap_max_depth + 1) + depth;

        controlled_sampler->seed(rng_index);
        controlled_sampler->start_iteration();

        auto res                     = trace_ray(rs);
        bootstrap_weights[rng_index] = res.first[1];
    }
};

std::tuple<size_t, double, void *, SamplingIntegrator::EvaluationCounters>
PSSMLTIntegrator::run_chain_scalar(size_t /*seed*/, size_t n_mutations,
                                   const Scene *scene, PCG32 &rng,
                                   ReplayableSampler *base_sampler,
                                   ImageBlock *block) {

    auto target_function = [this](const Array<float, 4> &values) {
        auto lum = values[1];
        if (m_tw_use_log_transform) {
            // Luminance, passed through a log transform.
            return enoki::max(m_background_energy + enoki::log(lum / m_estimated_average_lum + 1.f),
                              (Float) 0.f);
        } else if (m_tw_use_gamma_transform) {
            // Luminance, passed through a gamma transform.
            return enoki::pow(
                enoki::max(m_background_energy + lum, (Float) 0.f),
                m_tw_gamma_exponent);
        } else {
            // Luminance (original PSSMLT).
            return m_background_energy + lum;
        }
    };

    ScopedPhase sp(EProfilerPhase::EMLTRunChain);
    EvaluationCounters eval_counters;

    // Initialize local variables for selected random seed.
    MLTSampler *sampler = (MLTSampler *) base_sampler;
    RadianceSample3f rs(scene, sampler);

    /// Format: XYZAW.
    Array<Float, 4> v_current, v_proposed;
    Float luminance_current, luminance_proposed;
    Point2f p_current, p_proposed;
    size_t ls_count     = 0;
    double ls_luminance = 0.0;

    // Initial state
    sampler->start_iteration();

#if MTS_EXPORT_RAYS
    RayExporter::instance()->disable();
#endif
    std::tie(v_current, p_current) = trace_ray(rs);
    ++eval_counters[EScalarPath];
    luminance_current = target_function(v_current);
#if MTS_EXPORT_RAYS
    RayExporter::instance()->enable();
#endif

    // Initial state cannot have zero probability, otherwise bootstrap sampling
    // did not do its job correctly.
    Assert(luminance_current > 0.0f, "Bootstrap sampling failed.");

    // Run the Markov chain for the desired number of steps
    for (size_t j = 0; j < n_mutations && !should_stop(); ++j) {
#if MTS_EXPORT_RAYS
        RayExporter::instance()->start_group(
            tfm::format("%s-it-%06d", Thread::thread()->name(), j));
#endif
        sampler->start_iteration();
        std::tie(v_proposed, p_proposed) = trace_ray(rs);
        ++eval_counters[EScalarPath];
        luminance_proposed = target_function(v_proposed);

        // Accumulate luminance from large steps to recover overall scale
        if (sampler->is_large_step() && !enoki::isnan(v_proposed[1])) {
            ++ls_count;
            ls_luminance += (double) v_proposed[1];
        }

        // Compute acceptance probability for proposed sample
        Float accept =
            std::min(luminance_proposed / luminance_current, (Float) 1.0f);
        Assert(!std::isnan(accept));
        Assert(none_nested(enoki::isnan(v_proposed)) &&
               none_nested(enoki::isnan(v_current)));
        Assert(accept >= 0.0f);

        // Expected value technique: splat both current and proposed
        // samples, weighted by acceptance probability. The total amount of
        // luminance accumulated is always 1.
        if (accept > 0.0f) {
            Float weight = accept / luminance_proposed;
            Array<Float, 5> entry(
                weight * v_proposed[0], weight * v_proposed[1],
                weight * v_proposed[2], accept * v_proposed[3], accept);
            block->put(p_proposed, entry.data());
        }
        Float recip = 1.0f - accept;
        if (recip > 0.0f) {
            Float weight = recip / luminance_current;
            Array<Float, 5> entry(weight * v_current[0], weight * v_current[1],
                                  weight * v_current[2], recip * v_current[3],
                                  recip);
            block->put(p_current, entry.data());
        }

        // Accept or reject the proposal
        if (rng.next_float32() < accept) {
            v_current         = v_proposed;
            luminance_current = luminance_proposed;
            p_current         = p_proposed;
            sampler->accept();
        } else
            sampler->reject();

        if (j > 0 && (j % 30000) == 0)
            report_progress(30000);
        if (!m_callbacks.empty()) {
            // TODO: check performance impact of this call
            Float preview_correction = 1.0f;
            if (ls_count > 0 && ls_luminance > 0)
                preview_correction = ((Float) ls_luminance / ls_count);
            Integrator::notify(block->bitmap(), preview_correction);
        }
    }

    return std::make_tuple(ls_count, ls_luminance, nullptr, eval_counters);
};

void PSSMLTIntegrator::apply_mlt_normalization(Bitmap *storage,
                                               Float normalization) const {
    if (!enoki::isfinite(normalization) || normalization == 0.0f) {
        Log(EWarn, "Incorrect normalization factor: %f, setting to 1.0f",
            normalization);
        normalization = 1.0f;
    }
    Float *target        = (Float *) storage->data();
    size_t channel_count = storage->channel_count();
    size_t n_pixels      = storage->pixel_count();

    if (m_lum_map) {
        if (m_flat_image_path.size() > 0) {
            // For debugging, output the "flat" image as well.
            ref<Bitmap> flat_storage = new Bitmap(*storage);
            auto *flat               = (Float *) flat_storage->data();
            for (size_t i = 0; i < n_pixels; ++i) {
                flat[3] = normalization;
                flat[4] = normalization;
                flat += channel_count;
            }
            flat_storage->convert(Bitmap::EY, Struct::EFloat32, false)
                ->write(fs::path(m_flat_image_path));
            Log(EInfo, "Wrote flat image (debug) to: %s", m_flat_image_path);
        }

        /* Luminance map was used: we need to multiply back by the map's value
         * at each pixel. */
        Float *map = (Float *) m_lum_map->data();
        if (m_lum_map->pixel_count() == storage->pixel_count()) {
            for (size_t i = 0; i < n_pixels; ++i) {
                target[1] *= (*map + m_lum_map_epsilon);
                target[3] *=
                    normalization / (target[4] > 0.0f ? target[4] : 1.0f);
                target[4] = normalization;
                target += channel_count;
                map += 1;
            }
        } else {
            /* This function might be called with an image that includes a
             * border (for the reconstruction filter). In that case, we need to
             * make sure to get the indexing right. Values of the luminance map
             * in the border region are repeated with clamping. */
            int32_t border = storage->width() - m_lum_map->width();
            if (border < 0 || (border % 2) != 0 ||
                border != (int32_t)(storage->height() - m_lum_map->height()))
                Throw("Image (%s) and luminance map (%s) sizes do not match. "
                      "Could not infer border size (%d)",
                      storage->size(), m_lum_map->size(), border);
            border /= 2;

            for (int32_t y = 0; y < (int32_t) storage->height(); ++y) {
                int32_t oy = clamp(y - border, 0, m_lum_map->height() - 1);
                for (int32_t x = 0; x < (int32_t) storage->width(); ++x) {
                    int32_t ox = clamp(x - border, 0, m_lum_map->width() - 1);
                    Float lum  = *(map + oy * m_lum_map->width() + ox);

                    target[1] *= (lum + m_lum_map_epsilon);
                    target[3] *=
                        normalization / (target[4] > 0.0f ? target[4] : 1.0f);
                    target[4] = normalization;
                    target += channel_count;
                }
            }
        }

    } else {
        for (size_t i = 0; i < n_pixels; ++i) {
            /* Alpha should actually be divided by the splatting weight (flat
             * average of the values splatted). We further multiply by the new
             * normalization denominator so that it is unaffected by the
             * division that will come later. */
            target[3] *= normalization / (target[4] > 0.0f ? target[4] : 1.0f);
            target[4] = normalization; // Overwrite weight
            target += channel_count;
        }
    }
}

void PSSMLTIntegrator::add_direct_illumination(Scene *scene, bool vectorize) {
    auto *film   = scene->film();
    auto *bitmap = film->bitmap();

    /** Adds two XYZAW bitmaps, applying normalization weights separately.
     * `source` is added to `target`. */
    auto add_bitmaps = [](const Bitmap *source_, Bitmap *target_) {
        Assert(source_->pixel_format() == Bitmap::EXYZAW);
        Assert(target_->pixel_format() == Bitmap::EXYZAW);

        auto *source      = (Float *) source_->data();
        auto *target      = (Float *) target_->data();
        size_t n_channels = target_->channel_count();
        size_t n_pixels   = target_->pixel_count();
        Float tw, sw;
        for (size_t i = 0; i < n_pixels; ++i) {
            tw        = 1.0f / target[4];
            sw        = 1.0f / source[4];
            target[0] = fmadd(tw, target[0], sw * source[0]);
            target[1] = fmadd(tw, target[1], sw * source[1]);
            target[2] = fmadd(tw, target[2], sw * source[2]);
            target[3] = fmadd(tw, target[3], sw * source[3]);
            target[4] = 1;

            source += n_channels;
            target += n_channels;
        }
    };

    if (m_direct_pass_path == "none") {
        Log(EWarn, "Specified direct_pass_path = none, not adding direct "
                   "illumination");
        return;
    }
    if (m_direct_pass_image) {
        Log(EInfo, "Using provided direct illumination component.");
        add_bitmaps(m_direct_pass_image.get(), bitmap);
    } else {
        Log(EInfo, "Computing direct illumination component.");
        // Create a DirectIntegrator.
        Properties props("direct");
        props.set_long("shading_samples", m_direct_shading_samples);
        props.set_string("perf_counters_preset", "");
        m_sub_integrator =
            PluginManager::instance()->create_object<SamplingIntegrator>(props);

        // Copy what was saved to the Film so far.
        ref<Bitmap> indirect(new Bitmap(*bitmap));

        film->clear();
        m_sub_integrator->render(scene, vectorize);
        auto sub_counters = m_sub_integrator->evaluation_counters();
        m_eval_counters[EScalarDirect] += sub_counters[EScalarEvaluation];
        m_eval_counters[EVectorDirect] += sub_counters[EVectorEvaluation];

        /* Film now contains direct illumination, add back indirect component.
         * Note however that we must first apply the weights separately. */
        add_bitmaps(indirect.get(), bitmap);

        m_sub_integrator = nullptr;
    }
}

std::string PSSMLTIntegrator::to_string() const {
    using string::indent;

    std::ostringstream oss;
    oss << "PSSMLTIntegrator[" << std::endl
        << "  integrator = " << string::indent(m_integrator->to_string()) << ","
        << std::endl
        << "  bootstrap_samples_count = " << m_bootstrap_samples_count << ","
        << std::endl
        << "  bootstrap_max_depth = " << m_bootstrap_max_depth << ","
        << std::endl
        << "  chain_count = " << m_chain_count << "," << std::endl
        << "  mutations_per_pixel = " << m_mutations_per_pixel << ","
        << std::endl
        << "  sigma = " << m_sigma << "," << std::endl
        << "  large_step_probability = " << m_large_step_probability << ","
        << std::endl
        << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS(ReplayableSampler, Sampler);
MTS_IMPLEMENT_CLASS(MLTSampler, ReplayableSampler);
MTS_IMPLEMENT_CLASS(PSSMLTIntegrator, MonteCarloIntegrator);
NAMESPACE_END(mitsuba)
