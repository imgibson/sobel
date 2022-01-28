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

#include <immintrin.h> // Intel AVX-512

#define RSHIFT(shift) _mm512_permutexvar_ps(RShiftVec, shift)
#define LSHIFT(shift) _mm512_permutexvar_ps(LShiftVec, shift)
#define RSHIFTM(shift, merge) _mm512_mask_blend_ps(0b0000000000000001, RSHIFT(shift), _mm512_permutexvar_ps(RMergeVec, merge))
#define LSHIFTM(shift, merge) _mm512_mask_blend_ps(0b1000000000000000, LSHIFT(shift), _mm512_permutexvar_ps(LMergeVec, merge))

alignas(64) static constexpr uint32_t RShift[16] = { 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
alignas(64) static constexpr uint32_t LShift[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15 };
alignas(64) static constexpr uint32_t RMerge[16] = { 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 };
alignas(64) static constexpr uint32_t LMerge[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

static const float ScaleFactor = 1.0f / sqrtf(32.0f);

void sobel_filter_avx512(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytes_per_line_src, uint32_t bytes_per_line_dst)
{
	static const __m512i RShiftVec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&RShift[0]));
	static const __m512i LShiftVec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&LShift[0]));
	static const __m512i RMergeVec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&RMerge[0]));
	static const __m512i LMergeVec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&LMerge[0]));
	static const __m512 ScaleVec = _mm512_set1_ps(ScaleFactor);

	// Verify 512 bit alignment
	assert((reinterpret_cast<uintptr_t>(src) & 63u) == 0u);
	assert((reinterpret_cast<uintptr_t>(dst) & 63u) == 0u);
	assert((bytes_per_line_src & 63u) == 0u);
	assert((bytes_per_line_dst & 63u) == 0u);
	// Verify minimum SIMD width
	assert(width >= 16u);

	const float* pr = src;
	const float* cr = src;
	const float* nr = reinterpret_cast<const float*>(&reinterpret_cast<const uint8_t*>(src)[bytes_per_line_src]);
	const float* lr = reinterpret_cast<const float*>(&reinterpret_cast<const uint8_t*>(src)[(height - 1u) * static_cast<uintptr_t>(bytes_per_line_src)]);

	float* dr = dst;

	if ((width & 15u) == 0u)
	{
		if (width >= 32u)
		{
			while (pr < lr)
			{
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

				out = _mm512_mul_ps(_mm512_sqrt_ps(out), ScaleVec);

				_mm512_store_ps(dr, out);

				uint32_t x = 32u;

				for (; x < width; x += 16u)
				{
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

					out = _mm512_mul_ps(_mm512_sqrt_ps(out), ScaleVec);

					_mm512_store_ps(&dr[x - 16u], out);
				}

				xout = _mm512_sub_ps(RSHIFTM(nextx, currx), LSHIFT(nextx));
				xout = _mm512_mul_ps(xout, xout);

				out = _mm512_add_ps(_mm512_add_ps(nexty, nexty), _mm512_add_ps(RSHIFTM(nexty, curry), LSHIFT(nexty)));
				out = _mm512_fmadd_ps(out, out, xout);

				out = _mm512_mul_ps(_mm512_sqrt_ps(out), ScaleVec);

				_mm512_store_ps(&dr[x - 16u], out);

				pr = cr;
				cr = nr;
				nr = reinterpret_cast<const float*>(&reinterpret_cast<const uint8_t*>(nr)[bytes_per_line_src]);
				if (nr > lr)
					nr = lr;

				dr = reinterpret_cast<float*>(&reinterpret_cast<uint8_t*>(dr)[bytes_per_line_dst]);
			}
		}
		else
		{
			while (pr < lr)
			{
				const __m512 top = _mm512_load_ps(pr);
				const __m512 mid = _mm512_load_ps(cr);
				const __m512 low = _mm512_load_ps(nr);

				__m512 xout = _mm512_add_ps(_mm512_add_ps(mid, mid), _mm512_add_ps(top, low));
				xout = _mm512_sub_ps(RSHIFT(xout), LSHIFT(xout));
				xout = _mm512_mul_ps(xout, xout);

				__m512 out = _mm512_sub_ps(top, low);
				out = _mm512_add_ps(_mm512_add_ps(out, out), _mm512_add_ps(RSHIFT(out), LSHIFT(out)));
				out = _mm512_fmadd_ps(out, out, xout);

				out = _mm512_mul_ps(_mm512_sqrt_ps(out), ScaleVec);

				_mm512_store_ps(dr, out);

				pr = cr;
				cr = nr;
				nr = reinterpret_cast<const float*>(&reinterpret_cast<const uint8_t*>(nr)[bytes_per_line_src]);
				if (nr > lr)
					nr = lr;

				dr = reinterpret_cast<float*>(&reinterpret_cast<uint8_t*>(dr)[bytes_per_line_dst]);
			}
		}
	}
	else
	{
		if (width >= 32u)
		{
			const uint32_t count = 16u * (width / 16u);
			const uint32_t lx = width - 1u;

			while (pr < lr)
			{
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

				out = _mm512_mul_ps(_mm512_sqrt_ps(out), ScaleVec);

				_mm512_store_ps(dr, out);

				uint32_t x = 32u;

				for (; x < count; x += 16u)
				{
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

					out = _mm512_mul_ps(_mm512_sqrt_ps(out), ScaleVec);

					_mm512_store_ps(&dr[x - 16u], out);
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
				nr = reinterpret_cast<const float*>(&reinterpret_cast<const uint8_t*>(nr)[bytes_per_line_src]);
				if (nr > lr)
					nr = lr;

				dr = reinterpret_cast<float*>(&reinterpret_cast<uint8_t*>(dr)[bytes_per_line_dst]);
			}
		}
		else if (width >= 16u)
		{
			const uint32_t lx = width - 1u;

			while (pr < lr)
			{
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

					out = _mm512_mul_ps(_mm512_sqrt_ps(out), ScaleVec);

					_mm512_store_ps(dr, out);
				}

				for (uint32_t x = 15u; x < lx; ++x)
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
				nr = reinterpret_cast<const float*>(&reinterpret_cast<const uint8_t*>(nr)[bytes_per_line_src]);
				if (nr > lr)
					nr = lr;

				dr = reinterpret_cast<float*>(&reinterpret_cast<uint8_t*>(dr)[bytes_per_line_dst]);
			}
		}
	}
}
