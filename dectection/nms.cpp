#include <vector>
#include <algorithm>
#include <utility>

double iou(const std::vector<double>& boxA, const std::vector<double>& boxB)
{
    const double eps = 1e-6;
    double areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    double areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    double x1 = std::max(boxA[0], boxB[0]);
    double y1 = std::max(boxA[1], boxB[1]);
    double x2 = std::min(boxA[2], boxB[2]);
    double y2 = std::min(boxA[3], boxB[3]);
    double w = std::max(0.0, x2 - x1);
    double h = std::max(0.0, y2 - y1);
    double inter = w * h;
    return inter / (areaA + areaB - inter + eps);
}

void nms(std::vector<std::vector<double>>& boxes, const double iou_threshold)
{
    // Sort boxes by score (box[4] contains the score)
    std::sort(boxes.begin(), boxes.end(), [](const std::vector<double>& boxA, const std::vector<double>& boxB) {
        return boxA[4] > boxB[4];
    });

    std::vector<bool> keep(boxes.size(), true);  // Keep track of which boxes to retain

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (!keep[i])
            continue;

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (keep[j] && iou(boxes[i], boxes[j]) > iou_threshold)
            {
                keep[j] = false;  // Mark overlapping box to be discarded
            }
        }
    }

    // Collect valid boxes
    std::vector<std::vector<double>> result;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (keep[i])
        {
            result.push_back(boxes[i]);
        }
    }

    boxes = std::move(result);  // Update original boxes with the filtered result
}
