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

#include <cassert>
#include <cmath>
#include <cstdint>

#include <immintrin.h> // Intel AVX-512

#include "sobel_filter.h"

#define RSHIFT(shift) _mm512_permutexvar_ps(kRightShiftVec, shift)
#define LSHIFT(shift) _mm512_permutexvar_ps(kLeftShiftVec, shift)
#define RSHIFTM(shift, merge) _mm512_mask_blend_ps(0b0000000000000001, RSHIFT(shift), _mm512_permutexvar_ps(kRightMergeVec, merge))
#define LSHIFTM(shift, merge) _mm512_mask_blend_ps(0b1000000000000000, LSHIFT(shift), _mm512_permutexvar_ps(kLeftMergeVec, merge))

static constexpr uint32_t kSimdWidth = 16u;
static constexpr uint32_t kByteAlign = kSimdWidth * sizeof(float);
static constexpr uint32_t kMaskAlign = kByteAlign - 1u;

alignas(64) static constexpr uint32_t kRightShift[16] = {  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14 };
alignas(64) static constexpr uint32_t kLeftShift[16] = {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 15 };
alignas(64) static constexpr uint32_t kRightMerge[16] = { 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 };
alignas(64) static constexpr uint32_t kLeftMerge[16] = {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 };

static const __m512i kRightShiftVec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&kRightShift[0]));
static const __m512i kLeftShiftVec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&kLeftShift[0]));
static const __m512i kRightMergeVec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&kRightMerge[0]));
static const __m512i kLeftMergeVec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&kLeftMerge[0]));

static const float kScaleFactor = 1.0f / sqrtf(32.0f);
static const __m512 kScaleVec = _mm512_set1_ps(kScaleFactor);

static inline const float* offset_ptr(const void* ptr, uintptr_t byteOffset) noexcept {
#ifdef _DEBUG
	assert((reinterpret_cast<uintptr_t>(ptr) & kMaskAlign) == 0u);
	assert((byteOffset & kMaskAlign) == 0u);
#endif
	const void* offsetPtr = &static_cast<const uint8_t*>(ptr)[byteOffset];
	return static_cast<const float*>(offsetPtr);
}

static inline float* offset_ptr(void* ptr, uintptr_t byteOffset) noexcept {
#ifdef _DEBUG
	assert((reinterpret_cast<uintptr_t>(ptr) & kMaskAlign) == 0u);
	assert((byteOffset & kMaskAlign) == 0u);
#endif
	void* offsetPtr = &static_cast<uint8_t*>(ptr)[byteOffset];
	return static_cast<float*>(offsetPtr);
}

void sobel_filter_avx512(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytesPerLineSrc, uint32_t bytesPerLineDst) noexcept {
#ifdef _DEBUG
	// Verify 512 bit alignment
	assert((reinterpret_cast<uintptr_t>(src) & kMaskAlign) == 0u);
	assert((reinterpret_cast<uintptr_t>(dst) & kMaskAlign) == 0u);
	assert((bytesPerLineSrc & kMaskAlign) == 0u);
	assert((bytesPerLineDst & kMaskAlign) == 0u);
	// Verify minimum SIMD width
	assert(width >= kSimdWidth);
#endif

	const float* pr = src;
	const float* cr = src;
	const float* nr = offset_ptr(src, bytesPerLineSrc);
	const float* lr = offset_ptr(src, (height - 1u) * static_cast<uintptr_t>(bytesPerLineSrc));

	float* dr = dst;

	if ((width & 15u) == 0u) {
		if (width >= 32u) {
			while (pr < lr) {
				__m512 top = _mm512_load_ps(pr);
				__m512 mid = _mm512_load_ps(cr);
				__m512 low = _mm512_load_ps(nr);

				__m512 currx = _mm512_add_ps(_mm512_add_ps(mid, mid), _mm512_add_ps(top, low));
				__m512 curry = _mm512_sub_ps(top, low);

				top = _mm512_load_ps(&pr[16u]);
				mid = _mm512_load_ps(&cr[16u]);
				low = _mm512_load_ps(&nr[16u]);

				__m512 nextx = _mm512_add_ps(_mm512_add_ps(mid, mid), _mm512_add_ps(top, low));
				__m512 nexty = _mm512_sub_ps(top, low);

				__m512 xout = _mm512_sub_ps(RSHIFT(currx), LSHIFTM(currx, nextx));
				xout = _mm512_mul_ps(xout, xout);

				__m512 out = _mm512_add_ps(_mm512_add_ps(curry, curry), _mm512_add_ps(RSHIFT(curry), LSHIFTM(curry, nexty)));
				out = _mm512_fmadd_ps(out, out, xout);

				out = _mm512_mul_ps(_mm512_sqrt_ps(out), kScaleVec);

				_mm512_store_ps(dr, out);

				uint32_t x = 32u;

				for (; x < width; x += 16u) {
					const __m512 prevx = currx;
					const __m512 prevy = curry;

					currx = nextx;
					curry = nexty;

					top = _mm512_load_ps(&pr[x]);
					mid = _mm512_load_ps(&cr[x]);
					low = _mm512_load_ps(&nr[x]);

					nextx = _mm512_add_ps(_mm512_add_ps(mid, mid), _mm512_add_ps(top, low));
					nexty = _mm512_sub_ps(top, low);

					xout = _mm512_sub_ps(RSHIFTM(currx, prevx), LSHIFTM(currx, nextx));
					xout = _mm512_mul_ps(xout, xout);

					out = _mm512_add_ps(_mm512_add_ps(curry, curry), _mm512_add_ps(RSHIFTM(curry, prevy), LSHIFTM(curry, nexty)));
					out = _mm512_fmadd_ps(out, out, xout);

					out = _mm512_mul_ps(_mm512_sqrt_ps(out), kScaleVec);

					_mm512_store_ps(&dr[x - 16u], out);
				}

				xout = _mm512_sub_ps(RSHIFTM(nextx, currx), LSHIFT(nextx));
				xout = _mm512_mul_ps(xout, xout);

				out = _mm512_add_ps(_mm512_add_ps(nexty, nexty), _mm512_add_ps(RSHIFTM(nexty, curry), LSHIFT(nexty)));
				out = _mm512_fmadd_ps(out, out, xout);

				out = _mm512_mul_ps(_mm512_sqrt_ps(out), kScaleVec);

				_mm512_store_ps(&dr[x - 16u], out);

				pr = cr;
				cr = nr;
				nr = offset_ptr(nr, bytesPerLineSrc);
				if (nr > lr) {
					nr = lr;
				}
				dr = offset_ptr(dr, bytesPerLineDst);
			}
		} else {
			while (pr < lr) {
				const __m512 top = _mm512_load_ps(pr);
				const __m512 mid = _mm512_load_ps(cr);
				const __m512 low = _mm512_load_ps(nr);

				__m512 xout = _mm512_add_ps(_mm512_add_ps(mid, mid), _mm512_add_ps(top, low));
				xout = _mm512_sub_ps(RSHIFT(xout), LSHIFT(xout));
				xout = _mm512_mul_ps(xout, xout);

				__m512 out = _mm512_sub_ps(top, low);
				out = _mm512_add_ps(_mm512_add_ps(out, out), _mm512_add_ps(RSHIFT(out), LSHIFT(out)));
				out = _mm512_fmadd_ps(out, out, xout);

				out = _mm512_mul_ps(_mm512_sqrt_ps(out), kScaleVec);

				_mm512_store_ps(dr, out);

				pr = cr;
				cr = nr;
				nr = offset_ptr(nr, bytesPerLineSrc);
				if (nr > lr) {
					nr = lr;
				}
				dr = offset_ptr(dr, bytesPerLineDst);
			}
		}
	} else {
		if (width >= 32u) {
			const uint32_t count = 16u * (width / 16u);
			const uint32_t lx = width - 1u;

			while (pr < lr) {
				__m512 top = _mm512_load_ps(pr);
				__m512 mid = _mm512_load_ps(cr);
				__m512 low = _mm512_load_ps(nr);

				__m512 currx = _mm512_add_ps(_mm512_add_ps(mid, mid), _mm512_add_ps(top, low));
				__m512 curry = _mm512_sub_ps(top, low);

				top = _mm512_load_ps(&pr[16u]);
				mid = _mm512_load_ps(&cr[16u]);
				low = _mm512_load_ps(&nr[16u]);

				__m512 nextx = _mm512_add_ps(_mm512_add_ps(mid, mid), _mm512_add_ps(top, low));
				__m512 nexty = _mm512_sub_ps(top, low);

				__m512 xout = _mm512_sub_ps(RSHIFT(currx), LSHIFTM(currx, nextx));
				xout = _mm512_mul_ps(xout, xout);

				__m512 out = _mm512_add_ps(_mm512_add_ps(curry, curry), _mm512_add_ps(RSHIFT(curry), LSHIFTM(curry, nexty)));
				out = _mm512_fmadd_ps(out, out, xout);

				out = _mm512_mul_ps(_mm512_sqrt_ps(out), kScaleVec);

				_mm512_store_ps(dr, out);

				uint32_t x = 32u;

				for (; x < count; x += 16u) {
					const __m512 prevx = currx;
					const __m512 prevy = curry;

					currx = nextx;
					curry = nexty;

					top = _mm512_load_ps(&pr[x]);
					mid = _mm512_load_ps(&cr[x]);
					low = _mm512_load_ps(&nr[x]);

					nextx = _mm512_add_ps(_mm512_add_ps(mid, mid), _mm512_add_ps(top, low));
					nexty = _mm512_sub_ps(top, low);

					xout = _mm512_sub_ps(RSHIFTM(currx, prevx), LSHIFTM(currx, nextx));
					xout = _mm512_mul_ps(xout, xout);

					out = _mm512_add_ps(_mm512_add_ps(curry, curry), _mm512_add_ps(RSHIFTM(curry, prevy), LSHIFTM(curry, nexty)));
					out = _mm512_fmadd_ps(out, out, xout);

					out = _mm512_mul_ps(_mm512_sqrt_ps(out), kScaleVec);

					_mm512_store_ps(&dr[x - 16u], out);
				}

				for (; x < lx; ++x) {
					const float dx =
						1.0f * (pr[x + 1u] - pr[x - 1u]) +
						2.0f * (cr[x + 1u] - cr[x - 1u]) +
						1.0f * (nr[x + 1u] - nr[x - 1u]);

					const float dy =
						1.0f * (pr[x - 1u] - nr[x - 1u]) +
						2.0f * (pr[x] - nr[x]) +
						1.0f * (pr[x + 1u] - nr[x + 1u]);

					dr[x] = sqrtf(dx * dx + dy * dy) * kScaleFactor;
				}

				{
					const float dx =
						1.0f * (pr[lx] - pr[lx - 1u]) +
						2.0f * (cr[lx] - cr[lx - 1u]) +
						1.0f * (nr[lx] - nr[lx - 1u]);

					const float dy =
						1.0f * (pr[lx - 1u] - nr[lx - 1u]) +
						2.0f * (pr[lx] - nr[lx]) +
						1.0f * (pr[lx] - nr[lx]);

					dr[x] = sqrtf(dx * dx + dy * dy) * kScaleFactor;
				}

				pr = cr;
				cr = nr;
				nr = offset_ptr(nr, bytesPerLineSrc);
				if (nr > lr) {
					nr = lr;
				}
				dr = offset_ptr(dr, bytesPerLineDst);
			}
		} else if (width >= 16u) {
			const uint32_t lx = width - 1u;

			while (pr < lr) {
				{
					const __m512 top = _mm512_load_ps(pr);
					const __m512 mid = _mm512_load_ps(cr);
					const __m512 low = _mm512_load_ps(nr);

					const __m512 currx = _mm512_add_ps(_mm512_add_ps(mid, mid), _mm512_add_ps(top, low));
					const __m512 curry = _mm512_sub_ps(top, low);

					__m512 xout = _mm512_sub_ps(RSHIFT(currx), LSHIFT(currx));
					xout = _mm512_mul_ps(xout, xout);

					__m512 out = _mm512_add_ps(_mm512_add_ps(curry, curry), _mm512_add_ps(RSHIFT(curry), LSHIFT(curry)));
					out = _mm512_fmadd_ps(out, out, xout);

					out = _mm512_mul_ps(_mm512_sqrt_ps(out), kScaleVec);

					_mm512_store_ps(dr, out);
				}

				for (uint32_t x = 15u; x < lx; ++x) {
					const float dx =
						1.0f * (pr[x + 1u] - pr[x - 1u]) +
						2.0f * (cr[x + 1u] - cr[x - 1u]) +
						1.0f * (nr[x + 1u] - nr[x - 1u]);

					const float dy =
						1.0f * (pr[x - 1u] - nr[x - 1u]) +
						2.0f * (pr[x] - nr[x]) +
						1.0f * (pr[x + 1u] - nr[x + 1u]);

					dr[x] = sqrtf(dx * dx + dy * dy) * kScaleFactor;
				}

				{
					const float dx =
						1.0f * (pr[lx] - pr[lx - 1u]) +
						2.0f * (cr[lx] - cr[lx - 1u]) +
						1.0f * (nr[lx] - nr[lx - 1u]);

					const float dy =
						1.0f * (pr[lx - 1u] - nr[lx - 1u]) +
						2.0f * (pr[lx] - nr[lx]) +
						1.0f * (pr[lx] - nr[lx]);

					dr[lx] = sqrtf(dx * dx + dy * dy) * kScaleFactor;
				}

				pr = cr;
				cr = nr;
				nr = offset_ptr(nr, bytesPerLineSrc);
				if (nr > lr) {
					nr = lr;
				}
				dr = offset_ptr(dr, bytesPerLineDst);
			}
		}
	}
}
