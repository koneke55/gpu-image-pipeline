/* Minimal stub for stb_image - replace with upstream file for full support. */
#pragma once
extern "C" float* stbi_loadf(const char*, int*, int*, int*, int){return nullptr;}
extern "C" void stbi_image_free(void*){}
