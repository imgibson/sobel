/*!
 * Sobel Filter (the "software") provided by Anders Lind ("author") license agreements.
 * - This software is free for both personal and commercial use. You may install and use it on your computers free of charge.
 * - You may NOT modify, de-compile, disassemble or reverse engineer the software.
 * - You may use, copy, sell, redistribute or give the software to third part freely as long as the software is not modified.
 * - The software remains property of the authors also in case of dissemination to third parties.
 * - The software's name and logo are not to be used to identify other products or services.
 * - THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * - The authors reserve the rights to change the license agreements in future versions of the software
 */

#pragma once

#include <cstdint>

void sobel_filter(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytesPerLineSrc, uint32_t bytesPerLineDst) noexcept;
void sobel_filter_sse2(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytesPerLineSrc, uint32_t bytesPerLineDst) noexcept;
void sobel_filter_avx2(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytesPerLineSrc, uint32_t bytesPerLineDst) noexcept;
void sobel_filter_avx512(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytesPerLineSrc, uint32_t bytesPerLineDst) noexcept;
