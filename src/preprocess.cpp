#include "preprocess.hpp"

#include <nvtx3/nvtx3.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

cv::Size letterbox_info(cv::Size src, int size, LetterboxInfo &lb)
{
    lb.scale = std::min(size / static_cast<float>(src.width),
                        size / static_cast<float>(src.height));
    const int nw = static_cast<int>(std::round(src.width * lb.scale));
    const int nh = static_cast<int>(std::round(src.height * lb.scale));
    lb.pad_x = (size - nw) / 2;
    lb.pad_y = (size - nh) / 2;
    return {nw, nh};
}

void letterbox(const cv::Mat &src, cv::Mat &resized, cv::Mat &out, int size, LetterboxInfo &lb)
{
    const cv::Size rs = letterbox_info(src.size(), size, lb);
    const int nw = rs.width, nh = rs.height;

    cv::resize(src, resized, rs, 0, 0, cv::INTER_LINEAR);

    out.create(size, size, src.type()); // no-op after the first frame if size and type match
    out.setTo(cv::Scalar::all(114));    // memset, no allocation
    resized.copyTo(out(cv::Rect(lb.pad_x, lb.pad_y, nw, nh)));
}

void preprocess(Frame &frame, int size)
{
    cv::Mat resized, letterboxed;

    letterbox(frame.image, resized, letterboxed, size, frame.lb);
    cv::dnn::blobFromImage(letterboxed, frame.blob, 1.0 / 255.0, {size, size}, cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
}

NppPre::NppPre(cudaStream_t stream)
{
    check(nppGetStreamContext(&ctx));
    ctx.hStream = stream;
    check(cudaStreamGetFlags(stream, &ctx.nStreamFlags));
}

void NppPre::ensure(int sz)
{
    if (sz == size)
        return;
    size = sz;
    filled_for = {};
    const size_t box_px = static_cast<size_t>(sz) * sz;
    d_box = make_device<Npp8u>(box_px * 3);
    d_f32 = make_device<Npp32f>(box_px * 3);
}

void preprocess_npp(Frame &frame, int size, NppPre &npp)
{
    LetterboxInfo &lb = frame.lb;
    const cv::Size rs = letterbox_info(frame.image.size(), size, lb);
    const int nw = rs.width, nh = rs.height;

    npp.ensure(size);
    // The border is the same every frame, so it is laid down once. nppiCopyConstBorder_8u_C3R_Ctx
    // would redo it per frame, but in NPP 12.1 it ignores ctx.hStream and races the upload.
    if (rs != npp.filled_for)
    {
        check(cudaMemsetAsync(npp.d_box.get(), 114, static_cast<size_t>(size) * size * 3, npp.ctx.hStream));
        npp.filled_for = rs;
    }
    const int rs_step = nw * 3;
    const int box_step = size * 3;
    const int f32_step = size * 3 * sizeof(Npp32f);
    const NppiSize box_sz{size, size};

    cv::resize(frame.image, npp.resized, rs, 0, 0, cv::INTER_LINEAR);

    {
        nvtx3::scoped_range r{"h2d"};
        Npp8u *roi = npp.d_box.get() + static_cast<size_t>(lb.pad_y) * box_step + lb.pad_x * 3;
        check(cudaMemcpy2DAsync(roi, box_step, npp.resized.data, npp.resized.step, rs_step, nh,
                                cudaMemcpyHostToDevice, npp.ctx.hStream));
    }

    {
        nvtx3::scoped_range r{"npp"};
        const int bgr2rgb[3] = {2, 1, 0};
        check(nppiSwapChannels_8u_C3IR_Ctx(npp.d_box.get(), box_step, box_sz, bgr2rgb, npp.ctx));

        check(nppiConvert_8u32f_C3R_Ctx(npp.d_box.get(), box_step, npp.d_f32.get(), f32_step, box_sz, npp.ctx));

        const Npp32f inv255[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
        check(nppiMulC_32f_C3IR_Ctx(inv255, npp.d_f32.get(), f32_step, box_sz, npp.ctx));

        // HWC -> NCHW
        const size_t plane = static_cast<size_t>(size) * size;
        Npp32f *const planes[3] = {frame.d_in.get(), frame.d_in.get() + plane, frame.d_in.get() + 2 * plane};
        check(nppiCopy_32f_C3P3R_Ctx(npp.d_f32.get(), f32_step, planes, size * sizeof(Npp32f), box_sz, npp.ctx));

        check(cudaStreamSynchronize(npp.ctx.hStream)); // d_in is complete for any stream, and pre timing includes the GPU work
    }
}

void preprocess_cpu(Frame &frame, int size, cudaStream_t stream)
{
    preprocess(frame, size);
    check(cudaMemcpyAsync(frame.d_in.get(), frame.blob.data, frame.blob.total() * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
}
