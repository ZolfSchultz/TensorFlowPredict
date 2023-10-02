using System;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace TensorTest
{
    internal class ImageClassificator
    {
        //string dir = "label_image_data"; // общая папка
        //string pbFile = "inception_v3_2016_08_28_frozen.pb"; //файл модели
        int input_height = 299;
        int input_width = 299;
        int input_mean = 0;
        int input_std = 255;
       // string input_name = "import/input";
       // string output_name = "import/InceptionV3/Predictions/Reshape_1";



        public bool Run(string imagepath)
        {
            string picFile = Path.GetFileName(imagepath);
            string[]? labels = new string[31] { "Abstract", "Aerial", "Animals", "Black and White", "Boudoir", "Celebrities", "City and Architecture", "Commercial", "Concert", "Family", "Fashion", "Film", "Fine Art", "Food", "Journalism", "Landscapes", "Macro", "Nature", "Night", "Nude", "Other", "People", "Performing Arts", "Sport", "Still Life", "Street", "Transportation", "Travel", "Underwater", "Urban Exploration", "Wedding" };

            var nd = ReadTensorFromImageFile(imagepath,
                input_height: input_height,
                input_width: input_width,
                input_mean: input_mean,
                input_std: input_std);









            var graph = new Graph();
            graph.Import(@"Assets\category.pb");
            Operation[] ops = graph.ToArray();
            var input_operation = ops[0];
            var output_operation = ops[ops.Length - 1];

            NDArray results;
            var sess = tf.Session(graph);
            results = sess.run(output_operation.outputs[0],
                new FeedItem(input_operation.outputs[0], nd));

            results = np.squeeze(results);

            var argsort = np.argsort(results);
            var top_k = argsort.ToArray<int>()
                .Skip((int)results.size - 5)
                .Reverse()
                .ToArray();

            foreach (float idx in top_k)
                Console.WriteLine($"{ picFile}: {idx} {labels[(int)idx]}, {results[(int)idx]}");

            return true;
        }

        private NDArray ReadTensorFromImageFile(string file_name,
                                int input_height = 299,
                                int input_width = 299,
                                int input_mean = 0,
                                int input_std = 0)
        {
            var graph = tf.Graph().as_default();

            var file_reader = tf.io.read_file(file_name, "file_reader");
            var image_reader = tf.image.decode_jpeg(file_reader, channels: 3, name: "jpeg_reader");
            var caster = tf.cast(image_reader, tf.float32);
            var dims_expander = tf.expand_dims(caster, 0);
            var resize = tf.constant(new int[] { input_height, input_width });
            var bilinear = tf.image.resize_bilinear(dims_expander, resize);
            var sub = tf.subtract(bilinear, new float[] { input_mean });
            //var normalized = tf.divide(sub, new float[] { input_std });

            var sess = tf.Session(graph);
            return sess.run(sub);
        }



    }
}
