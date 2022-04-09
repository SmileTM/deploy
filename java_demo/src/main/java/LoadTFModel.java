import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;

import java.util.HashMap;
import java.util.Map;

public class LoadTFModel {
    public static void main(String[] args) {


        // 定义形状
        Shape shape = Shape.of(1, 10);
        TFloat32 x = TFloat32.tensorOf(shape);
        // Init 设置输入的向量
        x.set(NdArrays.vectorOf(1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f), 0);
        System.out.println("x 形状：" + x.shape());

        //定义输入向量
        //查看对应输入变量对应的 名称
        //saved_model_cli show --all --dir test
        Map<String, Tensor> inputs = new HashMap<>();
        inputs.put("input_1", x);
        inputs.put("input_2", x);

        // 加载模型 并预测
        SavedModelBundle model = SavedModelBundle.load("src/main/resources/test");

        Map<String, Tensor> result = model.call(inputs);

        System.out.println(result);

        // 得到输出 output_1
        // TFloat32 output_1 = (TFloat32) result.get("output_1");

        // 为了方便建议 在模型内部进行 logits 向label的转换工作
        TInt64 output_1 = (TInt64) result.get("output_1");
        System.out.println(output_1.shape());
//        System.out.println(output_1.getLong(0，1)); // 这里按坐标取值 取第0向量中第1个元素
        System.out.println(output_1.getLong(0)); // 这里按坐标取值

        long start = System.currentTimeMillis();

        for (int i = 0; i < 100000; i++) {
            model.call(inputs);
        }
        long end = System.currentTimeMillis();
        System.out.println((end - start) / 1000.);


    }
}
