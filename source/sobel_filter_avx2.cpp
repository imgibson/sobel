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

#include <immintrin.h> // Intel AVX

#include "sobel_filter.h"

#define RSHIFT(shift) _mm256_permutevar8x32_ps(shift, kRightShiftVec)
#define LSHIFT(shift) _mm256_permutevar8x32_ps(shift, kLeftShiftVec)
#define RSHIFTM(shift, merge) _mm256_blend_ps(RSHIFT(shift), _mm256_permutevar8x32_ps(merge, kRightMergeVec), 0b00000001)
#define LSHIFTM(shift, merge) _mm256_blend_ps(LSHIFT(shift), _mm256_permutevar8x32_ps(merge, kLeftMergeVec), 0b10000000)

static constexpr uint32_t kSimdWidth = 8u;
static constexpr uint32_t kByteAlign = kSimdWidth * sizeof(float);
static constexpr uint32_t kMaskAlign = kByteAlign - 1u;

alignas(32) static constexpr uint32_t kRightShift[8] = { 0, 0, 1, 2, 3, 4, 5, 6 };
alignas(32) static constexpr uint32_t kLeftShift[8] = { 1, 2, 3, 4, 5, 6, 7, 7 };
alignas(32) static constexpr uint32_t kRightMerge[8] = { 7, 7, 7, 7, 7, 7, 7, 7 };
alignas(32) static constexpr uint32_t kLeftMerge[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

static const __m256i kRightShiftVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&kRightShift[0]));
static const __m256i kLeftShiftVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&kLeftShift[0]));
static const __m256i kRightMergeVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&kRightMerge[0]));
static const __m256i kLeftMergeVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&kLeftMerge[0]));

static const float kScaleFactor = 1.0f / sqrtf(32.0f);
static const __m256 kScaleVec = _mm256_set1_ps(kScaleFactor);

static inline const float* offset_ptr(const void* ptr, uintptr_t byteOffset) {
#ifdef _DEBUG
	assert((reinterpret_cast<uintptr_t>(ptr) & kMaskAlign) == 0u);
	assert((byteOffset & kMaskAlign) == 0u);
#endif
	const void* offsetPtr = &static_cast<const uint8_t*>(ptr)[byteOffset];
	return static_cast<const float*>(offsetPtr);
}

static inline float* offset_ptr(void* ptr, uintptr_t byteOffset) {
#ifdef _DEBUG
	assert((reinterpret_cast<uintptr_t>(ptr) & kMaskAlign) == 0u);
	assert((byteOffset & kMaskAlign) == 0u);
#endif
	void* offsetPtr = &static_cast<uint8_t*>(ptr)[byteOffset];
	return static_cast<float*>(offsetPtr);
}

void sobel_filter_avx2(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytesPerLineSrc, uint32_t bytesPerLineDst) {
#ifdef _DEBUG
	// Verify 256 bit alignment
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

	if ((width & 7u) == 0u) {
		if (width >= 16u) {
			while (pr < lr) {
				__m256 top = _mm256_load_ps(pr);
				__m256 mid = _mm256_load_ps(cr);
				__m256 low = _mm256_load_ps(nr);

				__m256 currx = _mm256_add_ps(_mm256_add_ps(mid, mid), _mm256_add_ps(top, low));
				__m256 curry = _mm256_sub_ps(top, low);

				top = _mm256_load_ps(&pr[8u]);
				mid = _mm256_load_ps(&cr[8u]);
				low = _mm256_load_ps(&nr[8u]);

				__m256 nextx = _mm256_add_ps(_mm256_add_ps(mid, mid), _mm256_add_ps(top, low));
				__m256 nexty = _mm256_sub_ps(top, low);

				__m256 xout = _mm256_sub_ps(RSHIFT(currx), LSHIFTM(currx, nextx));
				xout = _mm256_mul_ps(xout, xout);

				__m256 out = _mm256_add_ps(_mm256_add_ps(curry, curry), _mm256_add_ps(RSHIFT(curry), LSHIFTM(curry, nexty)));
				out = _mm256_fmadd_ps(out, out, xout);

				out = _mm256_mul_ps(_mm256_sqrt_ps(out), kScaleVec);

				_mm256_store_ps(dr, out);

				uint32_t x = 16u;

				for (; x < width; x += 8u) {
					const __m256 prevx = currx;
					const __m256 prevy = curry;

					currx = nextx;
					curry = nexty;

					top = _mm256_load_ps(&pr[x]);
					mid = _mm256_load_ps(&cr[x]);
					low = _mm256_load_ps(&nr[x]);

					nextx = _mm256_add_ps(_mm256_add_ps(mid, mid), _mm256_add_ps(top, low));
					nexty = _mm256_sub_ps(top, low);

					xout = _mm256_sub_ps(RSHIFTM(currx, prevx), LSHIFTM(currx, nextx));
					xout = _mm256_mul_ps(xout, xout);

					out = _mm256_add_ps(_mm256_add_ps(curry, curry), _mm256_add_ps(RSHIFTM(curry, prevy), LSHIFTM(curry, nexty)));
					out = _mm256_fmadd_ps(out, out, xout);

					out = _mm256_mul_ps(_mm256_sqrt_ps(out), kScaleVec);

					_mm256_store_ps(&dr[x - 8u], out);
				}

				xout = _mm256_sub_ps(RSHIFTM(nextx, currx), LSHIFT(nextx));
				xout = _mm256_mul_ps(xout, xout);

				out = _mm256_add_ps(_mm256_add_ps(nexty, nexty), _mm256_add_ps(RSHIFTM(nexty, curry), LSHIFT(nexty)));
				out = _mm256_fmadd_ps(out, out, xout);

				out = _mm256_mul_ps(_mm256_sqrt_ps(out), kScaleVec);

				_mm256_store_ps(&dr[x - 8u], out);

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
				const __m256 top = _mm256_load_ps(pr);
				const __m256 mid = _mm256_load_ps(cr);
				const __m256 low = _mm256_load_ps(nr);

				__m256 xout = _mm256_add_ps(_mm256_add_ps(mid, mid), _mm256_add_ps(top, low));
				xout = _mm256_sub_ps(RSHIFT(xout), LSHIFT(xout));
				xout = _mm256_mul_ps(xout, xout);

				__m256 out = _mm256_sub_ps(top, low);
				out = _mm256_add_ps(_mm256_add_ps(out, out), _mm256_add_ps(RSHIFT(out), LSHIFT(out)));
				out = _mm256_fmadd_ps(out, out, xout);

				out = _mm256_mul_ps(_mm256_sqrt_ps(out), kScaleVec);

				_mm256_store_ps(dr, out);

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
		if (width >= 16u) {
			const uint32_t count = 8u * (width / 8u);
			const uint32_t lx = width - 1u;

			while (pr < lr) {
				__m256 top = _mm256_load_ps(pr);
				__m256 mid = _mm256_load_ps(cr);
				__m256 low = _mm256_load_ps(nr);

				__m256 currx = _mm256_add_ps(_mm256_add_ps(mid, mid), _mm256_add_ps(top, low));
				__m256 curry = _mm256_sub_ps(top, low);

				top = _mm256_load_ps(&pr[8u]);
				mid = _mm256_load_ps(&cr[8u]);
				low = _mm256_load_ps(&nr[8u]);

				__m256 nextx = _mm256_add_ps(_mm256_add_ps(mid, mid), _mm256_add_ps(top, low));
				__m256 nexty = _mm256_sub_ps(top, low);

				__m256 xout = _mm256_sub_ps(RSHIFT(currx), LSHIFTM(currx, nextx));
				xout = _mm256_mul_ps(xout, xout);

				__m256 out = _mm256_add_ps(_mm256_add_ps(curry, curry), _mm256_add_ps(RSHIFT(curry), LSHIFTM(curry, nexty)));
				out = _mm256_fmadd_ps(out, out, xout);

				out = _mm256_mul_ps(_mm256_sqrt_ps(out), kScaleVec);

				_mm256_store_ps(dr, out);

				uint32_t x = 16u;

				for (; x < count; x += 8u) {
					const __m256 prevx = currx;
					const __m256 prevy = curry;

					currx = nextx;
					curry = nexty;

					top = _mm256_load_ps(&pr[x]);
					mid = _mm256_load_ps(&cr[x]);
					low = _mm256_load_ps(&nr[x]);

					nextx = _mm256_add_ps(_mm256_add_ps(mid, mid), _mm256_add_ps(top, low));
					nexty = _mm256_sub_ps(top, low);

					xout = _mm256_sub_ps(RSHIFTM(currx, prevx), LSHIFTM(currx, nextx));
					xout = _mm256_mul_ps(xout, xout);

					out = _mm256_add_ps(_mm256_add_ps(curry, curry), _mm256_add_ps(RSHIFTM(curry, prevy), LSHIFTM(curry, nexty)));
					out = _mm256_fmadd_ps(out, out, xout);

					out = _mm256_mul_ps(_mm256_sqrt_ps(out), kScaleVec);

					_mm256_store_ps(&dr[x - 8u], out);
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
		} else if (width >= 8u) {
			const uint32_t lx = width - 1u;

			while (pr < lr) {
				{
					const __m256 top = _mm256_load_ps(pr);
					const __m256 mid = _mm256_load_ps(cr);
					const __m256 low = _mm256_load_ps(nr);

					const __m256 currx = _mm256_add_ps(_mm256_add_ps(mid, mid), _mm256_add_ps(top, low));
					const __m256 curry = _mm256_sub_ps(top, low);

					__m256 xout = _mm256_sub_ps(RSHIFT(currx), LSHIFT(currx));
					xout = _mm256_mul_ps(xout, xout);

					__m256 out = _mm256_add_ps(_mm256_add_ps(curry, curry), _mm256_add_ps(RSHIFT(curry), LSHIFT(curry)));
					out = _mm256_fmadd_ps(out, out, xout);

					out = _mm256_mul_ps(_mm256_sqrt_ps(out), kScaleVec);

					_mm256_store_ps(dr, out);
				}

				for (uint32_t x = 7u; x < lx; ++x) {
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
