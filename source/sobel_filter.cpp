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
#include <cmath>

static const float ScaleFactor = 1.0f / sqrtf(32.0f);

void sobel_filter(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytes_per_line_src, uint32_t bytes_per_line_dst)
{
	const float* pr = src;
	const float* cr = src;
	const float* nr = (const float*)((const uint8_t*)src + bytes_per_line_src);
	const float* lr = (const float*)((const uint8_t*)src + (height - 1u) * static_cast<uintptr_t>(bytes_per_line_src));

	float* dr = dst;

	const uint32_t lx = width - 1u;

	while (pr < lr)
	{
		{
			const float dx =
				1.0f * (pr[1u] - pr[0u]) +
				2.0f * (cr[1u] - cr[0u]) +
				1.0f * (nr[1u] - nr[0u]);

			const float dy =
				1.0f * (pr[0u] - nr[0u]) +
				2.0f * (pr[0u] - nr[0u]) +
				1.0f * (pr[1u] - nr[1u]);

			dr[0u] = sqrtf(dx * dx + dy * dy) * ScaleFactor;
		}

		for (uint32_t x = 1u; x < lx; ++x)
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
		nr = (const float*)((const uint8_t*)nr + bytes_per_line_src);
		if (nr > lr)
			nr = lr;

		dr = (float*)((uint8_t*)dr + bytes_per_line_dst);
	}
}
