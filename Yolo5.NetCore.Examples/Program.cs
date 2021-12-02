using System;
using System.Collections.Generic;
using System.Drawing;
using Yolo5.NetCore.Models;

namespace Yolo5.NetCore.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            using var image = Image.FromFile("input.jpg");
            using var yolo = new Yolo<YoloCocoModel>("Models/yolov5n6.onnx");
            var predictions = yolo.Predict(image);

            using var graphics = Graphics.FromImage(image);
            foreach (var prediction in predictions) // iterate predictions to draw results
            {
                var score = Math.Round(prediction.Score, 2);
                graphics.DrawRectangles(new Pen(Color.Blue, 1),
                    new[] { prediction.Rectangle });

                var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

                graphics.DrawString($"{prediction.Label.Name} ({score})",
                    new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(Color.White),
                    new PointF(x, y));
            }

            image.Save("output.jpg");
        }
    }
}
