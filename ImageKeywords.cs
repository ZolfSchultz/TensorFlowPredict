using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using Tensorflow;
using static Tensorflow.Binding;
using Tensorflow;
using Tensorflow.Keras;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Utils;
using System.IO;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Preprocessings;
using Tensorflow.Keras.Text;
using System.Collections;
using OneOf.Types;
using Tensorflow.Operations.Initializers;
using Razorvine.Pickle;
using System.IO;
using static Tensorflow.ApiDef.Types;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Text.RegularExpressions;
using static System.Net.Mime.MediaTypeNames;

namespace TensorTest
{


    internal class ImageKeywords
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
            
            var nd = ReadTensorFromImageFile(imagepath,
                input_height: input_height,
                input_width: input_width,
                input_mean: input_mean,
                input_std: input_std);









            var graph = new Graph();
            graph.Import(@"Assets\predict.pb");
            Operation[] ops = graph.ToArray();
            var input_operation = ops[0];
            var output_operation = ops[ops.Length - 1];

            NDArray results;
            var sess = tf.Session(graph);
            results = sess.run(output_operation.outputs[0],
                new FeedItem(input_operation.outputs[0], nd));


            //results = np.squeeze(results);
            var testing = keywords(results);
            results = np.squeeze(results);
            //var testing = keywords(results);
            //fetures = null;
            return true;
        }

        private NDArray ReadTensorFromImageFile(string file_name,
                                int input_height = 299,
                                int input_width = 299,
                                int input_mean = 0,
                                int input_std = 255)
        {
            var graph = tf.Graph().as_default();

            var file_reader = tf.io.read_file(file_name, "file_reader");
            var image_reader = tf.image.decode_jpeg(file_reader, channels: 3, name: "jpeg_reader");
            var caster = tf.cast(image_reader, tf.float32);
            var dims_expander = tf.expand_dims(caster, 0);
            var resize = tf.constant(new int[] { input_height, input_width });
            var bilinear = tf.image.resize_bilinear(dims_expander, resize);
            var sub = tf.subtract(bilinear, new float[] { input_mean });
            var normalized = tf.divide(sub, new float[] { input_std });

            var sess = tf.Session(graph);
            return sess.run(normalized);
        }

        public string keywords(NDArray nD)
        {
            //var graph2 = tf.Graph().as_default();
            //var tensor = tf.transpose(nD);
            //var sessions = tf.Session(graph2);
            //NDArray nDArray = sessions.run(tensor);

            // var model = keras.models.load_model(@"C:\Users\dmgus\Desktop\keywords.pb");
            //var pred = model.predict(nD);
            // var tensor = tf.transpose(nD);




            var index_word = File.ReadLines(@"Assets\index_word.csv").Select(line => line.Split(';')).ToDictionary(line => line[0], line => line[1]);
            var word_index = File.ReadLines(@"Assets\word_index.csv").Select(line => line.Split(';')).ToDictionary(line => line[0], line => line[1]);


            //string[] description  = File.ReadAllLines(@"Assets\tokenizer2.txt");
            string out_keyword = null;
            //List<string> test = new List<string>();


            //foreach (var line in description)
            //{
            //    test.Add(line);
            //}
            //test.Add(ebala);


            
            //var tokenizer = keras.preprocessing.text.Tokenizer(lower: false);

            //tokenizer.fit_on_texts(test);
            //List<string> in_text = new List<string>();

            string in_text = "startseq";
            //in_text.Add("startseq");
            
            int[] keydict = new[] { int.Parse(word_index[in_text]) };
            List<int[]>? int_sequence = new List<int[]>();
            int_sequence.Add(keydict);
            var graph = new Graph();
            graph.Import(@"Assets\keywords.pb");
            //var model = graph.Import(@"Assets\keywords.pb");
            Operation[] ops = graph.ToArray();
            var input_operation = ops[0]; // входной слой для признаков фотографии
            var input_operation2 = ops[1]; // входной слой для текста
            var output_operation = ops[ops.Length - 1];
            // var pred = model.predict((nD, seq));
            NDArray results;
            var sess = tf.Session(graph);
            for (int i = 1; i <= 169; i++)
            {
                //var tokenizer = new Tokenizer()
                //tokenizer.fit_on_texts(test);
                //var list = tokenizer.texts_to_sequences(in_text);
                Sequence sequence = new Sequence();
                var seq = sequence.pad_sequences(int_sequence, 169);

                seq = seq.astype(tf.float32);
               // seq = seq.astype(tf.float16);

                results = sess.run(output_operation.outputs[0], new FeedItem(input_operation.outputs[0], nD), new FeedItem(input_operation2.outputs[0], seq));
                //results = np.squeeze(results);
                // int argsort = np.argmax(results);
                var argmax = np.argmax(results, 1);
                string key = argmax[0].ToString();
                var word = index_word[key];
                if (word is None)
                {
                    break;
                }

                if (i == 1)
                {
                    out_keyword = in_text + " " + word;
                }
                else
                {
                    out_keyword = out_keyword + " " + word;
                }
                

                if (word == "endseq")
                {
                    break;
                }
                int_sequence.Clear();

                Array.Resize(ref keydict, keydict.Length + 1);
                keydict[keydict.Length - 1] = int.Parse(key);
                int_sequence.Add(keydict);

            }

            //out_keyword = out_keyword.Remove(0, 9);
            //out_keyword = out_keyword.Remove(out_keyword.Length-7);
            out_keyword = out_keyword.Replace(" ", ",");

            var res = Regex.Split(out_keyword, "(?=\\p{Lu})");

            out_keyword = null;
            foreach (var keys in res)
            {
                out_keyword = out_keyword + keys + " ";
            }

            out_keyword = out_keyword.Remove(0, 10);
            out_keyword = out_keyword.Remove(out_keyword.Length-8);
            Console.WriteLine(out_keyword);
            return out_keyword;
        }




    }
}

