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

void preprocess(FrameData &framedata, int size)
{
    cv::Mat resized, letterboxed;

    letterbox(framedata.frame, resized, letterboxed, size, framedata.lb);
    cv::dnn::blobFromImage(letterboxed, framedata.blob, 1.0 / 255.0, {size, size}, cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
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
    const size_t box_px = static_cast<size_t>(sz) * sz;
    d_resized = make_device<Npp8u>(box_px * 3); // resized fits inside the box
    d_box = make_device<Npp8u>(box_px * 3);
    d_f32 = make_device<Npp32f>(box_px * 3);
}

void preprocess_npp(FrameData &framedata, int size, NppPre &npp, Trt &trt)
{
    LetterboxInfo &lb = framedata.lb;
    const cv::Size rs = letterbox_info(framedata.frame.size(), size, lb);
    const int nw = rs.width, nh = rs.height;

    npp.ensure(size);
    const int rs_step = nw * 3;
    const int box_step = size * 3;
    const int f32_step = size * 3 * sizeof(Npp32f);
    const NppiSize box_sz{size, size};

    cv::resize(framedata.frame, npp.resized, rs, 0, 0, cv::INTER_LINEAR);

    {
        nvtx3::scoped_range r{"h2d"};
        check(cudaMemcpy2DAsync(npp.d_resized.get(), rs_step, npp.resized.data, npp.resized.step, rs_step, nh,
                                cudaMemcpyHostToDevice, trt.stream.get()));
    }

    {
        nvtx3::scoped_range r{"npp"};
        const Npp8u pad[3] = {114, 114, 114};
        check(nppiCopyConstBorder_8u_C3R_Ctx(npp.d_resized.get(), rs_step, {nw, nh},
                                             npp.d_box.get(), box_step, box_sz, lb.pad_y, lb.pad_x, pad, npp.ctx));

        const int bgr2rgb[3] = {2, 1, 0};
        check(nppiSwapChannels_8u_C3IR_Ctx(npp.d_box.get(), box_step, box_sz, bgr2rgb, npp.ctx));

        check(nppiConvert_8u32f_C3R_Ctx(npp.d_box.get(), box_step, npp.d_f32.get(), f32_step, box_sz, npp.ctx));

        const Npp32f inv255[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
        check(nppiMulC_32f_C3IR_Ctx(inv255, npp.d_f32.get(), f32_step, box_sz, npp.ctx));

        // HWC -> NCHW
        const size_t plane = static_cast<size_t>(size) * size;
        Npp32f *const planes[3] = {trt.d_in.get(), trt.d_in.get() + plane, trt.d_in.get() + 2 * plane};
        check(nppiCopy_32f_C3P3R_Ctx(npp.d_f32.get(), f32_step, planes, size * sizeof(Npp32f), box_sz, npp.ctx));

        check(cudaStreamSynchronize(trt.stream.get())); // so t.pre includes the GPU work
    }
}

void preprocess_cpu(FrameData &framedata, int size, Trt &trt)
{
    preprocess(framedata, size);
    check(cudaMemcpyAsync(trt.d_in.get(), framedata.blob.data, framedata.blob.total() * sizeof(float),
                          cudaMemcpyHostToDevice, trt.stream.get()));
    check(cudaStreamSynchronize(trt.stream.get()));
}
