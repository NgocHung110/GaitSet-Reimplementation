from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
# Đảm bảo import đúng hàm evaluation từ file evaluator của bạn
# Nếu evaluator.py nằm cùng cấp:
# from evaluator import evaluation
# Nếu evaluator.py nằm trong model/utils.py:
from model.utils import evaluation # Giả sử file evaluator.py đã được đổi tên/chuyển vào model/utils.py
                                   # hoặc bạn có một file model/utils.py chứa hàm evaluation này.
                                   # Quan trọng: Tên import phải khớp với vị trí file và tên hàm.

from config import conf # Đảm bảo file config.py tồn tại và có các cấu hình cần thiết

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()

m = initialization(conf, test=opt.cache)[0]

print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)
print('Transforming...')
time = datetime.now()
# transform sẽ trả về (feature, view, seq_type, label)
# Đảm bảo rằng data_loader của bạn trả về dữ liệu test nếu test_source không rỗng
test_data_package = m.transform('test', opt.batch_size)

if test_data_package is None or test_data_package[0] is None or test_data_package[0].shape[0] == 0:
    print("Lỗi: Không có dữ liệu test được transform. Kiểm tra lại data_loader và cấu hình dataset.")
    exit()

print('Evaluating...')
print(f"Test data shape: {test_data_package[0].shape}")
print(f"Number of test samples: {len(test_data_package[3])}")
print(f"Unique labels: {len(set(test_data_package[3]))}")
print(f"Unique views: {set(test_data_package[1])}")
print(f"Unique seq_types: {set(test_data_package[2])}")

# conf['data'] được truyền vào evaluation, nó chứa config['data']['dataset']
acc = evaluation(test_data_package, conf['data'])
print('Evaluation complete. Cost:', datetime.now() - time)

print('\n===Rank-1 Accuracy===')
try:
    if acc is not None and acc.shape[0] > 0:
        # Tên các điều kiện probe tương ứng với ảnh
        probe_condition_names = ['NM', 'BG', 'CL']
        condition_full_names = ['NM vs NM', 'BG vs NM', 'CL vs NM']

        
        for i, (short_name, full_name) in enumerate(zip(probe_condition_names, condition_full_names)):
            if acc.shape[1] > 0 and acc.shape[2] > 0:
                rank1_acc = acc[i, 0, 0, 0]
                if np.isnan(rank1_acc):
                    print(f'{short_name} vs NM (view 000 vs 000): No data or error')
                else:
                    print(f'{short_name} vs NM (view 000 vs 000): {rank1_acc:.2f}%')

        # Tính trung bình Rank-1
        if acc.shape[1] > 0 and acc.shape[2] > 0:
            valid_acc_rank1 = acc[~np.isnan(acc[:, 0, 0, 0]), 0, 0, 0]
            if len(valid_acc_rank1) > 0:
                mean_rank1 = np.mean(valid_acc_rank1)
                print(f'Mean Rank-1 (all conditions, 000 vs 000): {mean_rank1:.2f}%')
            else:
                print("Mean Rank-1: No valid results to average.")

    else:
        print("Không có kết quả accuracy để hiển thị.")

except Exception as e:
    print(f"Error printing results: {e}")
    import traceback
    traceback.print_exc()