import os
import shutil

if __name__ == "__main__":
    src_path_qxw = r"F:\M18\qxw"
    src_path_ylj = r"F:\M18\yulianjie"

    dst_path = r"F:\M18\求同存异"

    model_list = ["lcnet_sample", "ddr_sample"]

    for model_name in model_list:
        for score in [0, 1, 2]:
            score_list_qxw = os.listdir(os.path.join(src_path_qxw, model_name, str(score)))
            score_list_ylj = os.listdir(os.path.join(src_path_ylj, model_name, str(score)))

            for image_name in score_list_ylj:
                if image_name in score_list_qxw:
                    shutil.copy(
                        os.path.join(src_path_ylj, model_name, str(score), image_name),
                        os.path.join(dst_path, model_name, str(score), image_name)
                    )
                else:
                    shutil.copy(
                        os.path.join(src_path_ylj, model_name, str(score), image_name),
                        os.path.join(dst_path, model_name, "diff", image_name)
                    )
