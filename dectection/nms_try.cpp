#include <vector>
#include <algorithm>
#include <utility>
#include <random>
#include <iomanip>
#include <iostream>

double iou(const std::vector<double>& boxA, const std::vector<double>& boxB) {
    const double eps = 1e-6;
    double areaA = (boxA[2] - boxA[0]) * (boxA[3]- boxA[1]);
    double areaB = (boxB[2] - boxB[0]) * (boxB[3]- boxB[1]);
    double x1 = std::max(boxA[0], boxB[0]);
    double y1 = std::max(boxA[1], boxB[1]);
    double x2 = std::min(boxA[2], boxB[2]);
    double y2 = std::min(boxA[3], boxB[3]);
    double w = std::max(0.0, x2 - x1);
    double h = std::max(0.0, y2 - y1);
    double inter_area = w * h;
    return inter_area / (areaA + areaB - inter_area + eps);
}

void nms(std::vector<std::vector<double>>& boxes, const double iou_threshold) {
    // box:[top left x, top left y, bottom right x, bottom right y, score]
    std::sort(boxes.begin(), boxes.end(), [](const std::vector<double>& boxA, const std::vector<double>& boxB) {
        return boxA[4] > boxB[4];
    });

    std::vector<bool> keep(boxes.size(), true);
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!keep[i]) {
            continue;
        }
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (keep[j] && iou(boxes[i], boxes[j]) > iou_threshold) {
                keep[j] = false;
            }
        }
    }

    std::vector<std::vector<double>> result;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (keep[i]) {
            result.emplace_back(boxes[i]);
        }
    }

    boxes = std::move(result);
}

int main() {
    // 设置随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // 生成模拟的边界框数据
    std::vector<std::vector<double>> boxes;
    const int num_boxes = 10; // 生成10个边界框作为示例
    for (int i = 0; i < num_boxes; ++i) {
        double x1 = dis(gen) * 100; // 左上角的x坐标
        double y1 = dis(gen) * 100; // 左上角的y坐标
        double x2 = x1 + 10 + dis(gen) * 20; // 右下角的x坐标（确保宽度至少为10）
        double y2 = y1 + 10 + dis(gen) * 20; // 右下角的y坐标（确保高度至少为10）
        double score = dis(gen); // 随机分数
        boxes.push_back({x1, y1, x2, y2, score});
    }

    // 输出原始边界框数据
    std::cout << "Original boxes:" << std::endl;
    for (const auto& box : boxes) {
        std::cout << "[" << std::fixed << std::setprecision(2)
                  << box[0] << ", " << box[1] << ", " << box[2] << ", " << box[3] << ", " << box[4] << "]"
                  << std::endl;
    }

    // 应用NMS算法
    double iou_threshold = 0.3; // 设置IOU阈值
    nms(boxes, iou_threshold);

    // 输出NMS处理后的边界框数据
    std::cout << "\nBoxes after NMS (IOU threshold = " << iou_threshold << "):" << std::endl;
    for (const auto& box : boxes) {
        std::cout << "[" << std::fixed << std::setprecision(2)
                  << box[0] << ", " << box[1] << ", " << box[2] << ", " << box[3] << ", " << box[4] << "]"
                  << std::endl;
    }

    return 0;
}