import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.LongNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class LoadBertModel {
    public static void main(String[] args) {

        Shape shape = Shape.of(1, 512);
        long[] raw_data = new long[512];
        for (int i = 0; i < 512; i++) {
            raw_data[i] = 1;
        }
        LongNdArray x = NdArrays.vectorOf(raw_data);
        TInt64 data = TInt64.tensorOf(shape);
        data.set(x, 0);
        System.out.println(data.shape());

        Map<String, Tensor> inputs = new HashMap<>();
        inputs.put("input_1", data);
        inputs.put("input_2", data);
        inputs.put("input_3", data);

        //加载模型
        SavedModelBundle model = SavedModelBundle.load("src/main/resources/testBert");
        //推理
        Map<String, Tensor> result = model.call(inputs);
        //解析结果
        TFloat32 raw_out = (TFloat32) result.get("output_1");
        System.out.println(raw_out);
        System.out.println("raw_out shape：" + raw_out.shape());

        ArrayList<Float> res = new ArrayList<>();

        for (int i = 0; i < (int) raw_out.shape().asArray()[1]; i++) {
            res.add(raw_out.getFloat(0, i));
        }

        System.out.println("结果为：" + res);


        Long start = System.currentTimeMillis();
        for (int i = 0; i < 50; i++) {
            model.call(inputs);
        }
        Long end = System.currentTimeMillis();
        System.out.println("循环50次 耗时：");
        System.out.println((end - start) / 1000.);

    }
}
