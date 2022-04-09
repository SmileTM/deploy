import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public class LoadTFModelByDJL {
    public static void main(String[] args) throws TranslateException, MalformedModelException, ModelNotFoundException, IOException {


        // 利用djl加载模型
        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelUrls("src/main/resources/test")
                .optEngine("TensorFlow")
                .optProgress(new ProgressBar())
//                .optDevice(Device.gpu())
                .build();
        ZooModel<NDList, NDList> model1 = criteria.loadModel();
        Predictor<NDList, NDList> predictor = model1.newPredictor();
        System.out.println(predictor);

        //定义输入数据格式
        NDManager manager = NDManager.newBaseManager();
        float[] f = new float[]{1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
        NDList x_in = new NDList(manager.create(f).reshape(-1, 10), manager.create(f).reshape(-1, 10));
        NDList x_out = predictor.predict(x_in);

        System.out.println(x_out);
        System.out.println("===========");
        System.out.println("output_1 输出形状为：" + x_out.get("output_1").getShape());
        System.out.println("===========");
        System.out.println("预测结果为：" + x_out.get("output_1").toArray()[0]);


//        System.out.println(x_out.get(0));
//        System.out.println(x_out.get("output_1"));
//        System.out.println(x_out.get("output_1").toString());
//        System.out.println("=======");
//        System.out.println(x_out.get("output_1").max());
//        System.out.println("=======");
//        System.out.println(x_out.get("output_1").toFloatArray());
//        System.out.println(x_out.get("output_1").toArray()[0]);
//
//        System.out.println(Arrays.toString(x_out.get("output_1").toArray()));
//        System.out.println(x_out.get("output_1").argMax(1).toArray());
//        System.out.println(x_out.get("output_1").argMin(1).toArray()[0]);

        long start = System.currentTimeMillis();
        for (int i = 0; i < 100000; i++) {
            predictor.predict(x_in);
        }
        long end = System.currentTimeMillis();
        System.out.println((end - start) / 1000.);

    }
}
