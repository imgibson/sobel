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

#include "sobel_filter.h"

#include <cstdint>
#include <cassert>
#include <cmath>

#include <immintrin.h> // Intel AVX

#define RSHIFT(shift) _mm256_permutevar8x32_ps(shift, RShiftVec)
#define LSHIFT(shift) _mm256_permutevar8x32_ps(shift, LShiftVec)
#define RSHIFTM(shift, merge) _mm256_blend_ps(RSHIFT(shift), _mm256_permutevar8x32_ps(merge, RMergeVec), 0b00000001)
#define LSHIFTM(shift, merge) _mm256_blend_ps(LSHIFT(shift), _mm256_permutevar8x32_ps(merge, LMergeVec), 0b10000000)

static const float ScaleFactor = 1.0f / sqrtf(32.0f);

alignas(32) static constexpr uint32_t RShift[8] = { 0, 0, 1, 2, 3, 4, 5, 6 };
alignas(32) static constexpr uint32_t LShift[8] = { 1, 2, 3, 4, 5, 6, 7, 7 };
alignas(32) static constexpr uint32_t RMerge[8] = { 7, 7, 7, 7, 7, 7, 7, 7 };
alignas(32) static constexpr uint32_t LMerge[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

static inline const float* offset_float_ptr(const float* ptr, uintptr_t byte_offset) { return reinterpret_cast<const float*>(&reinterpret_cast<const uint8_t*>(ptr)[byte_offset]); }
static inline float* offset_float_ptr(float* ptr, uintptr_t byte_offset) { return reinterpret_cast<float*>(&reinterpret_cast<uint8_t*>(ptr)[byte_offset]); }

void sobel_filter_avx2(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytes_per_line_src, uint32_t bytes_per_line_dst)
{
	const __m256 ScaleVec = _mm256_set1_ps(ScaleFactor);
	const __m256i RShiftVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&RShift[0]));
	const __m256i LShiftVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&LShift[0]));
	const __m256i RMergeVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&RMerge[0]));
	const __m256i LMergeVec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&LMerge[0]));

	// Verify 256 bit alignment
	assert((reinterpret_cast<uintptr_t>(src) & 31u) == 0u);
	assert((reinterpret_cast<uintptr_t>(dst) & 31u) == 0u);
	assert((bytes_per_line_src & 31u) == 0u);
	assert((bytes_per_line_dst & 31u) == 0u);
	// Verify minimum SIMD width
	assert(width >= 8u);

	const float* pr = src;
	const float* cr = src;
	const float* nr = offset_float_ptr(src, bytes_per_line_src);
	const float* lr = offset_float_ptr(src, (height - 1u) * static_cast<uintptr_t>(bytes_per_line_src));

	float* dr = dst;

	if ((width & 7u) == 0u) // Even mulitple of SIMD size
	{
		if (width >= 16u)
		{
			while (pr < lr)
			{
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

				out = _mm256_mul_ps(_mm256_sqrt_ps(out), ScaleVec);

				_mm256_store_ps(dr, out);

				uint32_t x = 16u;

				for (; x < width; x += 8u)
				{
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

					out = _mm256_mul_ps(_mm256_sqrt_ps(out), ScaleVec);

					_mm256_store_ps(&dr[x - 8u], out);
				}

				xout = _mm256_sub_ps(RSHIFTM(nextx, currx), LSHIFT(nextx));
				xout = _mm256_mul_ps(xout, xout);

				out = _mm256_add_ps(_mm256_add_ps(nexty, nexty), _mm256_add_ps(RSHIFTM(nexty, curry), LSHIFT(nexty)));
				out = _mm256_fmadd_ps(out, out, xout);

				out = _mm256_mul_ps(_mm256_sqrt_ps(out), ScaleVec);

				_mm256_store_ps(&dr[x - 8u], out);

				pr = cr;
				cr = nr;
				nr = offset_float_ptr(nr, bytes_per_line_src);
				if (nr > lr)
					nr = lr;

				dr = offset_float_ptr(dr, bytes_per_line_dst);
			}
		}
		else
		{
			while (pr < lr)
			{
				const __m256 top = _mm256_load_ps(pr);
				const __m256 mid = _mm256_load_ps(cr);
				const __m256 low = _mm256_load_ps(nr);

				__m256 xout = _mm256_add_ps(_mm256_add_ps(mid, mid), _mm256_add_ps(top, low));
				xout = _mm256_sub_ps(RSHIFT(xout), LSHIFT(xout));
				xout = _mm256_mul_ps(xout, xout);

				__m256 out = _mm256_sub_ps(top, low);
				out = _mm256_add_ps(_mm256_add_ps(out, out), _mm256_add_ps(RSHIFT(out), LSHIFT(out)));
				out = _mm256_fmadd_ps(out, out, xout);

				out = _mm256_mul_ps(_mm256_sqrt_ps(out), ScaleVec);

				_mm256_store_ps(dr, out);

				pr = cr;
				cr = nr;
				nr = offset_float_ptr(nr, bytes_per_line_src);
				if (nr > lr)
					nr = lr;

				dr = offset_float_ptr(dr, bytes_per_line_dst);
			}
		}
	}
	else
	{
		if (width >= 16u)
		{
			const uint32_t count = 8u * (width / 8u);
			const uint32_t lx = width - 1u;

			while (pr < lr)
			{
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

				out = _mm256_mul_ps(_mm256_sqrt_ps(out), ScaleVec);

				_mm256_store_ps(dr, out);

				uint32_t x = 16u;

				for (; x < count; x += 8u)
				{
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

					out = _mm256_mul_ps(_mm256_sqrt_ps(out), ScaleVec);

					_mm256_store_ps(&dr[x - 8u], out);
				}

				for (; x < lx; ++x)
				{
					const float dx =
						1.0f * (pr[x + 1u] - pr[x - 1u]) +
						2.0f * (cr[x + 1u] - cr[x - 1u]) +
						1.0f * (nr[x + 1u] - nr[x - 1u]);

					const float dy =
						1.0f * (pr[x - 1u] - nr[x - 1u]) +
						2.0f * (pr[x] - nr[x]) +
						1.0f * (pr[x + 1u] - nr[x + 1u]);

					dr[x] = sqrtf(dx * dx + dy * dy) * ScaleFactor;
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

					dr[x] = sqrtf(dx * dx + dy * dy) * ScaleFactor;
				}

				pr = cr;
				cr = nr;
				nr = offset_float_ptr(nr, bytes_per_line_src);
				if (nr > lr)
					nr = lr;

				dr = offset_float_ptr(dr, bytes_per_line_dst);
			}
		}
		else if (width >= 8u)
		{
			const uint32_t lx = width - 1u;

			while (pr < lr)
			{
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

					out = _mm256_mul_ps(_mm256_sqrt_ps(out), ScaleVec);

					_mm256_store_ps(dr, out);
				}

				for (uint32_t x = 7u; x < lx; ++x)
				{
					const float dx =
						1.0f * (pr[x + 1u] - pr[x - 1u]) +
						2.0f * (cr[x + 1u] - cr[x - 1u]) +
						1.0f * (nr[x + 1u] - nr[x - 1u]);

					const float dy =
						1.0f * (pr[x - 1u] - nr[x - 1u]) +
						2.0f * (pr[x] - nr[x]) +
						1.0f * (pr[x + 1u] - nr[x + 1u]);

					dr[x] = sqrtf(dx * dx + dy * dy) * ScaleFactor;
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

					dr[lx] = sqrtf(dx * dx + dy * dy) * ScaleFactor;
				}

				pr = cr;
				cr = nr;
				nr = offset_float_ptr(nr, bytes_per_line_src);
				if (nr > lr)
					nr = lr;

				dr = offset_float_ptr(dr, bytes_per_line_dst);
			}
		}
	}
}
