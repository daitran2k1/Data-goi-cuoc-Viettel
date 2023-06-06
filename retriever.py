"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


#Define our Cross-Encoder
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_stsbenchmark-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#We use phobert-base as base model and set num_labels=1, which predicts a continous score between 0 and 1
model = CrossEncoder('vinai/phobert-base', num_labels=1)


# Read STSb dataset
logger.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []

with open('prompt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
  
trainval_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(trainval_data, test_size=0.2, random_state=42)
for sample in train_data:
    train_samples.append(InputExample(texts=[sample[0], sample[1]], label=1))
    train_samples.append(InputExample(texts=[sample[1], sample[0]], label=1))
for sample in val_data:
    dev_samples.append(InputExample(texts=[sample[0], sample[1]], label=1))
for sample in test_data:
    test_samples.append(InputExample(texts=[sample[0], sample[1]], label=1))


# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##### Load model and eval on test set
model = CrossEncoder(model_save_path)

evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='sts-test')
evaluator(model)

# import numpy as np

# # We want to compute the similarity between the query sentence
# query = 'giá gói cước ST70 là bao nhiêu.'

# # With all sentences in the corpus
# corpus = ['A man is eating food.',
#           "gói cước ST70, 70000đ / 30 ngày (Đối với thuê bao trả sau: Đăng ký gói từ ngày 21 đến cuối tháng, phí gói còn 35000đ / tháng và giữ nguyên 1GB / ngày). Ưu đãi: Có ngay 30GB (1GB / ngày) Hết lưu lượng ngừng truy cập. Gói cước gia hạn khi hết chu kỳ. Đối tượng áp dụng: Thuê bao di động trả trước và trả sau theo danh sách. Đăng ký: Bấm Đăng ký / Bấm *098*70000# Hủy gia hạn: Bấm Hủy / Soạn tin HUY gửi 191 (Xác nhận Y gửi 191), ưu đãi còn lại của gói cước sẽ được bảo lưu và sử dụng đến hết chu kỳ. Hủy dịch vụ: Soạn tin HUYDATA gửi 191 (Xác nhận Y gửi 191), ưu đãi còn lại của gói cước sẽ không được bảo lưu.",
#           "gói cước ST90, 90000đ / chu kỳ (Đối với thuê bao trả sau: Đăng ký từ ngày 21 đến cuối tháng sẽ giảm 50% phí gói và có 1GB / ngày lưu lượng data tốc độ cao) Ưu đãi: 1GB / ngày data tốc độ cao, hết 1GB ngừng truy cập. Gói cước tự động gia hạn khi hết chu kỳ (Không bảo lưu data còn lại khi hết 24h hàng ngày) Ưu đãi sử dụng trong 30 ngày (trả trước), hết tháng (trả sau). Đăng ký: Bấm Đăng ký / Soạn ST90K gửi 191 / Bấm gọi *098*6789#.|Hủy gia hạn: Bấm Hủy hoặc soạn HUY gửi 191. Hủy gói: Soạn tin HUYDATA gửi 191.",
#           "gói cước ST90SV, Đối tượng đăng ký: Học sinh, Sinh viên hòa mạng mới (mua SIM mới) từ ngày 10-8-2018 trở đi. Phạm vi áp dụng: Toàn quốc. Thời gian triển khai: Từ 10-8-2018 đến khi có thông báo mới từ Viettel. Phí đăng ký: 70000đ / tháng. Ưu đãi gói cước ST90SV Viettel. Tặng 2GB lưu lượng data tốc độ cao (4G / 3G) mỗi ngày để phục vụ nhu cầu truy cập mạng, áp dụng điều đặn mỗi ngày trong chu kỳ gói. Không cộng dồn nếu dùng không hết. Sau khi hết dung lượng miễn phí sẽ ngắt truy cập mạng.",
#           "gói cước T1, 3000đ / ngày Ưu đãi: Truy cập không giới hạn lưu lượng ứng dụng TikTok đến 24h. Gói cước gia hạn hàng ngày. Đăng ký: Soạn T1 gửi 191 hoặc bấm gọi *098*1007# Hủy gia hạn: Soạn HUY T1 gửi 191 Hủy gói: Soạn HUYDATA T1 gửi 191",
#           "gói cước T100, Miễn phí 1000 phút gọi nội mạng và 50 phút gọi ngoại mạng. Thời gian hưởng khuyến mại: 12 tháng. Phí tham gia tháng đầu tiên (đã bao gồm VAT) = phí tham gia tháng / số ngày trong tháng * số ngày còn lại trong tháng. Lưu lượng ưu đãi tháng đăng ký đầu tiên = lưu lượng của gói / số ngày trong tháng * số ngày sử dụng còn lại của tháng. Khi hết số phút gọi khuyến mãi, cước gọi được tính như sau: Cước gọi Cước thuê bao tháng: 0đ / tháng, gọi nội mạng Viette: 890đ / phút (6s đầu 89đ và mỗi giây tiếp theo 14.83đ). Gọi ngoại mạng Viettel: 990đ / phút (block 6s đầu 99đ và mỗi giây tiếp theo 16.50đ). Gọi tới đầu số 069: 693đ / phút (block 6s đầu 69.3đ và mỗi giây tiếp theo 11.55đ). Cước nhắn tin SMS nội mạng trong nước: 300đ / bản tin, SMS ngoại mạng trong nước: 350đ / bản tin, SMS quốc tế: 2500đ / bản tin, MMS: 300đ / bản tin.",
#           "gói cước T30, 30000đ / 30 ngày Ưu đãi: Truy cập không giới hạn lưu lượng ứng dụng TikTok. Gói cước gia hạn sau 30 ngày. Đăng ký: Soạn T30 gửi 191 hoặc bấm gọi *098*1009#. Hủy gia hạn: Soạn HUY T30 gửi 191 Huy gói: Soạn HUYDATA T30 gửi 191",
#           "gói cước TD30, Miễn Phí 10GB data tốc độ cao dùng trong khoảng thời gian từ 23h đếm đến 6h sáng hôm sau, trong vong 30 ngày. Cước phí đăng ký: 30000đ / lần đăng ký. Thời gian sử dụng của gói cước: 30 ngày.",
#           "gói cước TGG90, 90000đ / 30 ngày có ưu đãi dùng tại Tiền Giang gồm: 30 phút gọi ngoại mạng, miễn phí 20 phút / cuộc gọi nội mạng, 4GB / ngày. Ưu đãi dùng ngoài tỉnh Tiền Giang: 2GB / 30 ngày."
#           ]

# # So we create the respective sentence combinations
# sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

# # Compute the similarity scores for these combinations
# similarity_scores = model.predict(sentence_combinations)

# # Sort the scores in decreasing order
# sim_scores_argsort = reversed(np.argsort(similarity_scores))

# # Print the scores
# print("Query:", query)
# for idx in sim_scores_argsort:
#     print("{:.2f}\t{}".format(similarity_scores[idx], corpus[idx]))