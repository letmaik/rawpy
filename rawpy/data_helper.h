#include <libraw/libraw.h>
#include <libraw/libraw_types.h>

// subset of libraw_colordata_t
typedef struct
{
  unsigned cblack[4102];
  unsigned black;
} libraw_colordata_black_level_t;

libraw_colordata_black_level_t adjust_bl_(LibRaw* libraw) {

  libraw_colordata_black_level_t C;
  memcpy(&C.cblack, &libraw->imgdata.rawdata.color.cblack, sizeof(C.cblack));
  C.black = libraw->imgdata.rawdata.color.black;

  unsigned filters = libraw->imgdata.idata.filters;

  // The following was copied unchanged from libraw_cxx.cpp LibRaw::adjust_bl().

  // Add common part to cblack[] early
  if (filters > 1000 && (C.cblack[4] + 1) / 2 == 1 &&
    (C.cblack[5] + 1) / 2 == 1)
  {
    int clrs[4];
    int lastg = -1, gcnt = 0;
    for (int c = 0; c < 4; c++)
    {
      clrs[c] = libraw->FC(c / 2, c % 2);
      if (clrs[c] == 1)
      {
        gcnt++;
        lastg = c;
      }
    }
    if (gcnt > 1 && lastg >= 0)
      clrs[lastg] = 3;
    for (int c = 0; c < 4; c++)
      C.cblack[clrs[c]] +=
        C.cblack[6 + c / 2 % C.cblack[4] * C.cblack[5] + c % 2 % C.cblack[5]];
    C.cblack[4] = C.cblack[5] = 0;
    // imgdata.idata.filters = sfilters;
  }
  else if (filters <= 1000 && C.cblack[4] == 1 &&
           C.cblack[5] == 1) // Fuji RAF dng
  {
    for (int c = 0; c < 4; c++)
      C.cblack[c] += C.cblack[6];
    C.cblack[4] = C.cblack[5] = 0;
  }
  // remove common part from C.cblack[]
  int i = C.cblack[3];
  int c;
  for (c = 0; c < 3; c++)
    if (i > (int)C.cblack[c])
      i = C.cblack[c];

  for (c = 0; c < 4; c++)
    C.cblack[c] -= i; // remove common part
  C.black += i;

  // Now calculate common part for cblack[6+] part and move it to C.black

  if (C.cblack[4] && C.cblack[5])
  {
    i = C.cblack[6];
    for (c = 1; c < int(C.cblack[4] * C.cblack[5]); c++)
      if (i > int(C.cblack[6 + c]))
        i = C.cblack[6 + c];
    // Remove i from cblack[6+]
    int nonz = 0;
    for (c = 0; c < int(C.cblack[4] * C.cblack[5]); c++)
    {
      C.cblack[6 + c] -= i;
      if (C.cblack[6 + c])
        nonz++;
    }
    C.black += i;
    if (!nonz)
      C.cblack[4] = C.cblack[5] = 0;
  }
  for (c = 0; c < 4; c++)
    C.cblack[c] += C.black;

  return C;
}