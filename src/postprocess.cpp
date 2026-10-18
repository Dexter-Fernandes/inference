#include "postprocess.hpp"

#include <opencv2/dnn.hpp>

#include <algorithm>

std::vector<Detection> postprocess(const cv::Mat &out, int num_boxes, const LetterboxInfo &lb, const cv::Size &orig, const Config &cfg)
{
    // YOLOV5 output: [1, num_boxes, 5 + num_classes]
    // row = cx, cy, w, h,  objectness, class scores ...
    // Only the first num_boxes rows are valid.
    const int stride = out.size[2];
    const int num_cls = stride - 5;
    const auto *data = reinterpret_cast<const float *>(out.data);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < num_boxes; ++i)
    {
        const float *row = data + static_cast<size_t>(i) * stride;
        const float obj = row[4];
        if (obj < cfg.conf_thr)
            continue;

        const float *cls_begin = row + 5;
        const float *best = std::max_element(cls_begin, cls_begin + num_cls);
        const int cls = static_cast<int>(best - cls_begin);
        const float conf = obj * (*best);
        if (conf < cfg.conf_thr)
            continue;
        if (!cfg.keep_classes.empty() && !cfg.keep_classes.count(cls))
            continue;

        // Undo letterbox
        const float x1 = (row[0] - row[2] / 2.f - lb.pad_x) / lb.scale;
        const float y1 = (row[1] - row[3] / 2.f - lb.pad_y) / lb.scale;
        const float x2 = (row[0] + row[2] / 2.f - lb.pad_x) / lb.scale;
        const float y2 = (row[1] + row[3] / 2.f - lb.pad_y) / lb.scale;

        cv::Rect box(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                     cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        box &= cv::Rect(0, 0, orig.width, orig.height);
        if (box.area() <= 0)
            continue;

        boxes.push_back(box);
        scores.push_back(conf);
        class_ids.push_back(cls);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxesBatched(boxes, scores, class_ids,
                             cfg.conf_thr, cfg.nms_thr, keep);

    std::vector<Detection> dets;
    dets.reserve(keep.size());
    for (int idx : keep)
        dets.push_back({class_ids[idx], scores[idx], boxes[idx]});
    return dets;
}

void postprocess_stage(FrameData &framedata, const Config &config)
{
    framedata.dets = postprocess(framedata.out, framedata.num_kept, framedata.lb, framedata.frame.size(), config);
}
