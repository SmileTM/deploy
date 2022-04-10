import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.LongNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.proto.framework.ConfigProto;
import org.tensorflow.proto.framework.GPUOptions;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class LoadBertModel {
    public static void main(String[] args) {

        Shape shape = Shape.of(1, 512);
        long[] raw_data1 = new long[512];
        long[] raw_data2 = new long[512];
        long[] raw_data3 = new long[512];
        for (int i = 0; i < 512; i++) {
            raw_data1[i] = 1;
            raw_data2[i] = 1;
            raw_data3[i] = 1;
        }
        LongNdArray x1 = NdArrays.vectorOf(raw_data1);
        LongNdArray x2 = NdArrays.vectorOf(raw_data2);
        LongNdArray x3 = NdArrays.vectorOf(raw_data3);
        TInt64 data1 = TInt64.tensorOf(shape);
        TInt64 data2 = TInt64.tensorOf(shape);
        TInt64 data3 = TInt64.tensorOf(shape);
        data1.set(x1, 0);
        data2.set(x2, 0);
        data3.set(x3, 0);
        System.out.println(data1.shape());

        Map<String, Tensor> inputs = new HashMap<>();
        inputs.put("input_1", data1);
        inputs.put("input_2", data2);
        inputs.put("input_3", data3);

        // 指定GPU 以及设置 auto memery graph


        GPUOptions gpu = GPUOptions.newBuilder() //
                .setVisibleDeviceList("6")
                .setPerProcessGpuMemoryFraction(0.8) //
                .setAllowGrowth(true) //
                .build(); //

        ConfigProto configProto = ConfigProto.newBuilder() //
                .setAllowSoftPlacement(true) //
                .setLogDevicePlacement(true) //
                .mergeGpuOptions(gpu) //
                .build(); //

        //加载模型
//        SavedModelBundle model = SavedModelBundle.load("src/main/resources/testBert");
        SavedModelBundle model = SavedModelBundle
                .loader("src/main/resources/testBert")
                .withConfigProto(configProto)
                .load();

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
        for (int i = 0; i < 100; i++) {

            shape = Shape.of(1, 512);
            raw_data1 = new long[512];
            raw_data2 = new long[512];
            raw_data3 = new long[512];
            for (int j = 0; j < 512; j++) {
                raw_data1[j] = i;
                raw_data2[j] = i;
                raw_data3[j] = i;
            }
            x1 = NdArrays.vectorOf(raw_data1);
            x2 = NdArrays.vectorOf(raw_data2);
            x3 = NdArrays.vectorOf(raw_data3);
            data1 = TInt64.tensorOf(shape);
            data2 = TInt64.tensorOf(shape);
            data3 = TInt64.tensorOf(shape);
            data1.set(x1, 0);
            data2.set(x2, 0);
            data3.set(x3, 0);
            System.out.println(data1.shape());
            inputs.clear();
            inputs.put("input_1", data1);
            inputs.put("input_2", data2);
            inputs.put("input_3", data3);
            model.call(inputs);

        }
        Long end = System.currentTimeMillis();
        System.out.println("循环100次 耗时：");
        System.out.println((end - start) / 1000.);


    }
}
