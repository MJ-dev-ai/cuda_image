#ifdef _WIN32
#define STBI_WINDOWS_UTF8
#define STBIW_WINDOWS_UTF8
#endif
// Emit stb implementation symbols in exactly one translation unit.
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

