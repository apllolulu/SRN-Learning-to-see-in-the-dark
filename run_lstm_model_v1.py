import os
import argparse
import tensorflow as tf
import models.model_v1 as model

"""
train : python run_lstm_model_v1.py --phase=train --batch_size=1 --lr=1e-4 --epoch=1000   --model=lstm --architecture=ResnetBlock
test:   python run_lstm_model_v1.py --phase=test

"""

def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--phase', type=str, default='train', help='determine whether train or test')
    parser.add_argument('--model', type=str, default='lstm', help='model type: [lstm | gray | color |SE ]')
    parser.add_argument('--architecture', type=str, default='ResnetBlock', help='architecture type: [ResnetBlock | SE_ResNeXt]')
    parser.add_argument('--batch_size', help='training batch size', type=int, default=1)
    parser.add_argument('--epoch', help='training epoch number', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=1024,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1024,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')


    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # 创建模型
    deblur = model.DEBLUR(args)

    if args.phase == 'test':
        #deblur.test(args.height, args.width, args.input_path, args.output_path)
        deblur.test(args.height, args.width)
    if args.phase == 'train':
        deblur.train()
    else:
        print('phase should be set to either test or train')


if __name__ == '__main__':
    tf.app.run()
