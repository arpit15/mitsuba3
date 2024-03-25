import mitsuba as mi
mi.set_variant('scalar_rgb')

import matplotlib.pyplot as plt

from ipdb import set_trace

logger = mi.Thread.thread().logger()
logger.set_log_level(mi.LogLevel.Debug)

model_fn = "tutorials/scenes/cbox.xml"

scene = mi.load_file(model_fn, integrator="sppm")
img = mi.render(scene, spp=1)

mi.util.write_bitmap("cbox.exr", img)

set_trace()
# plt.imshow(img)
# plt.show()