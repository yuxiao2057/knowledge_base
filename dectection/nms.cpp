#include <opencv2/opencv.hpp>  
#include <vector>  
#include <algorithm>  
  
void nms(std::vector<cv::Rect>& input_boxes, float threshold, std::vector<cv::Rect>& output_boxes)  
{  
    std::sort(input_boxes.begin(), input_boxes.end(), [](const cv::Rect& a, const cv::Rect& b)  
    {  
        return a.area() > b.area();  
    });  
  
    while (input_boxes.size() > 0)  
    {  
        cv::Rect box1 = input_boxes[0];  
        output_boxes.push_back(box1);  
  
        input_boxes.erase(input_boxes.begin());  
  
        for (auto it = input_boxes.begin(); it != input_boxes.end(); )  
        {  
            cv::Rect box2 = *it;  
  
            float iou = (box1 & box2).area() / (float)(box1.area() + box2.area() - (box1 & box2).area());  
  
            if (iou > threshold)  
                it = input_boxes.erase(it);  
            else  
                ++it;  
        }  
    }  
}  
