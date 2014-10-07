#include "libraw_config.h"

/* libraw only defines these to 1, not to 0, but for Cython we
   need a definition in either case */

#ifndef LIBRAW_USE_DEMOSAIC_PACK_GPL2
#define LIBRAW_USE_DEMOSAIC_PACK_GPL2 0
#endif

#ifndef LIBRAW_USE_DEMOSAIC_PACK_GPL3
#define LIBRAW_USE_DEMOSAIC_PACK_GPL3 0
#endif