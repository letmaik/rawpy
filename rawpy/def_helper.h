#include <libraw/libraw_version.h>
#include <libraw/libraw_const.h>

// xtrans supported since 0.16.0
#ifndef LIBRAW_XTRANS
#define LIBRAW_XTRANS 9
#endif

// the following define's are prepended with _ as they are either new or have different semantics

// libraw_config.h only exists since 0.15.4, and only in cmake builds
// hence all values defined below are irrelevant if < 0.15.4 or cmake wasn't used
#if _HAS_LIBRAW_CONFIG_H
#include "libraw_config.h"
#define _LIBRAW_HAS_FLAGS 1
#else
#define _LIBRAW_HAS_FLAGS 0
#endif

/* libraw only defines these to 1, not to 0, but for Cython we
   need a definition in either case */

#ifdef LIBRAW_USE_DNGLOSSYCODEC
#define _LIBRAW_USE_DNGLOSSYCODEC 1
#else
#define _LIBRAW_USE_DNGLOSSYCODEC 0
#endif

#ifdef LIBRAW_USE_DNGDEFLATECODEC
#define _LIBRAW_USE_DNGDEFLATECODEC 1
#else
#define _LIBRAW_USE_DNGDEFLATECODEC 0
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

#ifdef LIBRAW_USE_X3FTOOLS
#define _LIBRAW_USE_X3FTOOLS 1
#else
#define _LIBRAW_USE_X3FTOOLS 0
#endif

#ifdef LIBRAW_USE_6BY9RPI
#define _LIBRAW_USE_6BY9RPI 1
#else
#define _LIBRAW_USE_6BY9RPI 0
#endif
