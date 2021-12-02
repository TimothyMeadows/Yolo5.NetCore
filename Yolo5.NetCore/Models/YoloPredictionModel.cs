using System.Drawing;

namespace Yolo5.NetCore.Models
{
    public class YoloPredictionModel
    {
        public YoloLabelModel Label { get; set; }
        public RectangleF Rectangle { get; set; }
        public float Score { get; set; }

        public YoloPredictionModel(YoloLabelModel label, float confidence) : this(label)
        {
            Score = confidence;
        }

        public YoloPredictionModel(YoloLabelModel label)
        {
            Label = label;
        }
    }
}
