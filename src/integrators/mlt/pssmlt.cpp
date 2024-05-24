#include <mitsuba/render/mlt/mlt.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * Primary Sample Space MLT integrator plugin.
 *
 * Simply uses all methods implemented in the parent class, \ref
 * PSSMLTIntegrator.
 */
class PSSMLTIntegratorImpl final : public PSSMLTIntegrator {
public:
    explicit PSSMLTIntegratorImpl(const Properties &props)
        : PSSMLTIntegrator(props) {}

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS(PSSMLTIntegratorImpl, PSSMLTIntegrator);
MTS_EXPORT_PLUGIN(PSSMLTIntegratorImpl, "PSSMLT integrator");
NAMESPACE_END(mitsuba)
