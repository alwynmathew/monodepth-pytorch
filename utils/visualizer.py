
import numpy as np
import matplotlib.pyplot as plt
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
import csv
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):

        self.opt = opt
        self.use_html = opt.use_html
        self.tf_log = opt.isTrain and not opt.use_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
        util.mkdir(self.log_dir)

        # if using tensorboard
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        # if using simple html page
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            util.mkdirs([self.web_dir, self.img_dir])

        # save test image results
        if not opt.isTrain:
            self.test_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_img_out')
            util.mkdirs([self.test_dir])

        # log txt file head
        self.log_name_txt = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name_txt, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        # log csv file head
        header = ['epoch', 'iters', 'time', 'loss_G', 'loss_D']
        self.log_name_csv = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.csv')
        with open(self.log_name_csv, "w") as log_file:
            writer = csv.writer(log_file, delimiter=',')
            writer.writerow(header)

        # log csv file head for individual loss

        self.log_individual_csv = os.path.join(opt.checkpoints_dir, opt.name, 'loss_individual_log.csv')
        with open(self.log_individual_csv, "w") as log_file:
            writer = csv.writer(log_file, delimiter=',')
            writer.writerow(header)

        # save loss graph as png
        self.error_plot = os.path.join(opt.checkpoints_dir, opt.name, 'error_plot.png')
        self.individual_error_plot = os.path.join(opt.checkpoints_dir, opt.name, 'individual_error_plot.png')

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                # print(label)
                if label == 'left_disp' or label == 'right_disp' or label == 'left_disp_ref' or label == 'right_disp_ref' or label == 'disp_out':
                    util.save_image_(image_numpy, img_path)
                elif label == 'edge_map':
                    util.save_image__(image_numpy, img_path)
                else:
                    util.save_image(image_numpy, img_path)

            # update website
            if self.opt.isTrain:
                webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
                for n in range(epoch, 0, -1):
                    webpage.add_header('Epoch [%d] steps [%d] updated at [%s]' % (n, step, time.ctime()))
                    ims = []
                    txts = []
                    links = []

                    for label, image_numpy in visuals.items():

                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                    if len(ims) < 3:
                        webpage.add_images(ims, txts, links, width=self.win_size)
                    else:
                        num = len(ims)/2
                        j=0
                        for i in range(num):
                            start = j
                            end = j+2
                            webpage.add_images(ims[start:end], txts[start:end], links[start:end], width=self.win_size)
                            j+=2
                webpage.save()

    def display_test_results(self, index, visuals):
        for label, image_numpy in visuals.items():
            img_path = os.path.join(self.test_dir, '%s_%.3d.jpg' % (label, index))
            util.save_image_(image_numpy[:,:,1], img_path)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        log_data = [epoch, i, t]
        for k, v in errors.items():
            message += '%s: %s ' % (k, v)
            log_data.append(v[0])
            if self.opt.headstart != -1:
                log_data.append(0)
        # print epoch iter error to console
        print(message)

        # write to csv file
        with open(self.log_name_csv, "a") as log_file:
            writer = csv.writer(log_file, delimiter=',')
            writer.writerow(log_data)
        # write to txt log file
        with open(self.log_name_txt, "a") as log_file:
            log_file.write('%s\n' % message)

        # plot error in graph and save figure
        index = []
        loss_D_list = []
        loss_G_list = []
        with open(self.log_name_csv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                index.append(int(row['iters']))
                loss_G_list.append(float(row['loss_G']))
                loss_D_list.append(float(row['loss_D']))

        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.plot(index, loss_G_list, 'g', label="loss G")
        plt.plot(index, loss_D_list, 'r', label="loss D")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.savefig(self.error_plot)
        plt.close()


    # plot induvidual error
    def plot_individual_loss(self, loss):
        with open(self.log_individual_csv, "a") as log_file:
            writer = csv.writer(log_file, delimiter=',')
            if self.opt.headstart != -1:
                # loss+=[0,0,0]
                loss+=[0,0]
            writer.writerow(loss)
        im_loss = []
        lr_loss = []
        disp_loss = []
        # G_advloss = []
        D_realloss = []
        D_fakeloss = []
        with open(self.log_individual_csv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # index.append(int(row['iters']))
                im_loss.append(float(row['im_loss'] .strip('[]')))
                lr_loss.append(float(row['lr_loss'] .strip('[]')))
                disp_loss.append(float(row['disp_loss'] .strip('[]')))
                if self.opt.headstart == -1:
                    # G_advloss.append(float(row['G_advloss'] .strip('[]')))
                    D_realloss.append(float(row['D_realloss'] .strip('[]')))
                    D_fakeloss.append(float(row['D_fakeloss'] .strip('[]')))
                else:
                    # G_advloss.append(float(row['G_advloss']))
                    D_realloss.append(float(row['D_realloss']))
                    D_fakeloss.append(float(row['D_fakeloss']))
        
        plt.figure()        
        plt.xlabel("iteration")
        plt.ylabel("error")
        index1 = range(0,len(im_loss))
        plt.plot(index1, im_loss, 'g', label="im_loss")
        plt.plot(index1, lr_loss, 'r', label="lr_loss")
        plt.plot(index1, disp_loss, 'c', label="disp_loss")
        # plt.plot(index1, G_advloss, 'm', label="G_advloss")
        plt.plot(index1, D_realloss, 'k', label="D_realloss")
        plt.plot(index1, D_fakeloss, 'y', label="D_fakeloss")
        # if len(im_loss) == 1:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
        plt.savefig(self.individual_error_plot)
        plt.close()
        # plt.clf()

    # adv loss only
    def plot_individual_loss_adv(self, loss):
        with open(self.log_individual_csv, "a") as log_file:
            writer = csv.writer(log_file, delimiter=',')
            if self.opt.headstart != -1:
                # loss+=[0,0,0]
                loss+=[0,0]
            writer.writerow(loss)
        l1 = []
        # G_advloss = []
        D_realloss = []
        D_fakeloss = []
        with open(self.log_individual_csv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # index.append(int(row['iters']))
                l1.append(float(row['l1'] .strip('[]')))
                if self.opt.headstart == -1:
                    # G_advloss.append(float(row['G_advloss'] .strip('[]')))
                    D_realloss.append(float(row['D_realloss'] .strip('[]')))
                    D_fakeloss.append(float(row['D_fakeloss'] .strip('[]')))
                else:
                    # G_advloss.append(float(row['G_advloss']))
                    D_realloss.append(float(row['D_realloss']))
                    D_fakeloss.append(float(row['D_fakeloss']))
        
        plt.figure()        
        plt.xlabel("iteration")
        plt.ylabel("error")
        index1 = range(0,len(l1))
        plt.plot(index1, l1, 'g', label="l1")
        # plt.plot(index1, G_advloss, 'm', label="G_advloss")
        plt.plot(index1, D_realloss, 'k', label="D_realloss")
        plt.plot(index1, D_fakeloss, 'y', label="D_fakeloss")
        # if len(im_loss) == 1:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
        plt.savefig(self.individual_error_plot)
        plt.close()

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
