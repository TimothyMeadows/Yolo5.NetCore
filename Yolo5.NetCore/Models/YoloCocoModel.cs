using System.Collections.Generic;

namespace Yolo5.NetCore.Models
{
    public class YoloCocoModel : YoloModel
    {
        public override int Width { get; set; } = 1280;
        public override int Height { get; set; } = 1280;
        public override int Depth { get; set; } = 3;

        public override int Dimensions { get; set; } = 85;

        public override int[] Strides { get; set; } = new int[] { 8, 16, 32, 64 };

        public override int[][][] Anchors { get; set; } = new int[][][]
        {
            new int[][] { new int[] { 019, 027 }, new int[] { 044, 040 }, new int[] { 038, 094 } },
            new int[][] { new int[] { 096, 068 }, new int[] { 086, 152 }, new int[] { 180, 137 } },
            new int[][] { new int[] { 140, 301 }, new int[] { 303, 264 }, new int[] { 238, 542 } },
            new int[][] { new int[] { 436, 615 }, new int[] { 739, 380 }, new int[] { 925, 792 } }
        };

        public override int[] Shapes { get; set; } = new int[] { 160, 80, 40, 20 };

        public override float Confidence { get; set; } = 0.20f;
        public override float MulConfidence { get; set; } = 0.25f;
        public override float Overlap { get; set; } = 0.45f;

        public override string[] Outputs { get; set; } = new[] { "output" };

        public override List<YoloLabelModel> Labels { get; set; } = new List<YoloLabelModel>()
        {
            new YoloLabelModel { Id = 1, Name = "person" },
            new YoloLabelModel { Id = 2, Name = "bicycle" },
            new YoloLabelModel { Id = 3, Name = "car" },
            new YoloLabelModel { Id = 4, Name = "motorcycle" },
            new YoloLabelModel { Id = 5, Name = "airplane" },
            new YoloLabelModel { Id = 6, Name = "bus" },
            new YoloLabelModel { Id = 7, Name = "train" },
            new YoloLabelModel { Id = 8, Name = "truck" },
            new YoloLabelModel { Id = 9, Name = "boat" },
            new YoloLabelModel { Id = 10, Name = "traffic light" },
            new YoloLabelModel { Id = 11, Name = "fire hydrant" },
            new YoloLabelModel { Id = 12, Name = "stop sign" },
            new YoloLabelModel { Id = 13, Name = "parking meter" },
            new YoloLabelModel { Id = 14, Name = "bench" },
            new YoloLabelModel { Id = 15, Name = "bird" },
            new YoloLabelModel { Id = 16, Name = "cat" },
            new YoloLabelModel { Id = 17, Name = "dog" },
            new YoloLabelModel { Id = 18, Name = "horse" },
            new YoloLabelModel { Id = 19, Name = "sheep" },
            new YoloLabelModel { Id = 20, Name = "cow" },
            new YoloLabelModel { Id = 21, Name = "elephant" },
            new YoloLabelModel { Id = 22, Name = "bear" },
            new YoloLabelModel { Id = 23, Name = "zebra" },
            new YoloLabelModel { Id = 24, Name = "giraffe" },
            new YoloLabelModel { Id = 25, Name = "backpack" },
            new YoloLabelModel { Id = 26, Name = "umbrella" },
            new YoloLabelModel { Id = 27, Name = "handbag" },
            new YoloLabelModel { Id = 28, Name = "tie" },
            new YoloLabelModel { Id = 29, Name = "suitcase" },
            new YoloLabelModel { Id = 30, Name = "frisbee" },
            new YoloLabelModel { Id = 31, Name = "skis" },
            new YoloLabelModel { Id = 32, Name = "snowboard" },
            new YoloLabelModel { Id = 33, Name = "sports ball" },
            new YoloLabelModel { Id = 34, Name = "kite" },
            new YoloLabelModel { Id = 35, Name = "baseball bat" },
            new YoloLabelModel { Id = 36, Name = "baseball glove" },
            new YoloLabelModel { Id = 37, Name = "skateboard" },
            new YoloLabelModel { Id = 38, Name = "surfboard" },
            new YoloLabelModel { Id = 39, Name = "tennis racket" },
            new YoloLabelModel { Id = 40, Name = "bottle" },
            new YoloLabelModel { Id = 41, Name = "wine glass" },
            new YoloLabelModel { Id = 42, Name = "cup" },
            new YoloLabelModel { Id = 43, Name = "fork" },
            new YoloLabelModel { Id = 44, Name = "knife" },
            new YoloLabelModel { Id = 45, Name = "spoon" },
            new YoloLabelModel { Id = 46, Name = "bowl" },
            new YoloLabelModel { Id = 47, Name = "banana" },
            new YoloLabelModel { Id = 48, Name = "apple" },
            new YoloLabelModel { Id = 49, Name = "sandwich" },
            new YoloLabelModel { Id = 50, Name = "orange" },
            new YoloLabelModel { Id = 51, Name = "broccoli" },
            new YoloLabelModel { Id = 52, Name = "carrot" },
            new YoloLabelModel { Id = 53, Name = "hot dog" },
            new YoloLabelModel { Id = 54, Name = "pizza" },
            new YoloLabelModel { Id = 55, Name = "donut" },
            new YoloLabelModel { Id = 56, Name = "cake" },
            new YoloLabelModel { Id = 57, Name = "chair" },
            new YoloLabelModel { Id = 58, Name = "couch" },
            new YoloLabelModel { Id = 59, Name = "potted plant" },
            new YoloLabelModel { Id = 60, Name = "bed" },
            new YoloLabelModel { Id = 61, Name = "dining table" },
            new YoloLabelModel { Id = 62, Name = "toilet" },
            new YoloLabelModel { Id = 63, Name = "tv" },
            new YoloLabelModel { Id = 64, Name = "laptop" },
            new YoloLabelModel { Id = 65, Name = "mouse" },
            new YoloLabelModel { Id = 66, Name = "remote" },
            new YoloLabelModel { Id = 67, Name = "keyboard" },
            new YoloLabelModel { Id = 68, Name = "cell phone" },
            new YoloLabelModel { Id = 69, Name = "microwave" },
            new YoloLabelModel { Id = 70, Name = "oven" },
            new YoloLabelModel { Id = 71, Name = "toaster" },
            new YoloLabelModel { Id = 72, Name = "sink" },
            new YoloLabelModel { Id = 73, Name = "refrigerator" },
            new YoloLabelModel { Id = 74, Name = "book" },
            new YoloLabelModel { Id = 75, Name = "clock" },
            new YoloLabelModel { Id = 76, Name = "vase" },
            new YoloLabelModel { Id = 77, Name = "scissors" },
            new YoloLabelModel { Id = 78, Name = "teddy bear" },
            new YoloLabelModel { Id = 79, Name = "hair drier" },
            new YoloLabelModel { Id = 80, Name = "toothbrush" }
        };

        public override bool UseDetect { get; set; } = true;

        public YoloCocoModel()
        {

        }
    }
}
