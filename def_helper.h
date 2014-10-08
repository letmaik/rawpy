#include "libraw_config.h"

/* libraw only defines these to 1, not to 0, but for Cython we
   need a definition in either case */

#ifdef LIBRAW_USE_DNGLOSSYCODEC
#define _LIBRAW_USE_DNGLOSSYCODEC 1
#else
#define _LIBRAW_USE_DNGLOSSYCODEC 0
#endif

#ifdef LIBRAW_USE_OPENMP
#define _LIBRAW_USE_OPENMP 1
#else
#define _LIBRAW_USE_OPENMP 0
#endif

#ifdef LIBRAW_USE_LCMS
#define _LIBRAW_USE_LCMS 1
#else
#define _LIBRAW_USE_LCMS 0
#endif

#ifdef LIBRAW_USE_REDCINECODEC
#define _LIBRAW_USE_REDCINECODEC 1
#else
#define _LIBRAW_USE_REDCINECODEC 0
#endif

#ifdef LIBRAW_USE_RAWSPEED
#define _LIBRAW_USE_RAWSPEED 1
#else
#define _LIBRAW_USE_RAWSPEED 0
#endif

#ifdef LIBRAW_USE_DEMOSAIC_PACK_GPL2
#define _LIBRAW_USE_DEMOSAIC_PACK_GPL2 1
#else
#define _LIBRAW_USE_DEMOSAIC_PACK_GPL2 0
#endif

#ifdef LIBRAW_USE_DEMOSAIC_PACK_GPL3
#define _LIBRAW_USE_DEMOSAIC_PACK_GPL3 1
#else
#define _LIBRAW_USE_DEMOSAIC_PACK_GPL3 0
#endif
