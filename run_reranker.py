import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from librerank.utils import *
from librerank.reranker import *
from librerank.rl_reranker import *
from librerank.CMR_generator import *
from librerank.CMR_evaluator import *
import datetime
import numpy as np

from pfevaluator.pfevaluator.distribution import Metric
from pfevaluator.pfevaluator.root import Root


import wandb

# import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.set_visible_devices(gpus[2], 'GPU')
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)
# # Set the visible GPUs
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Select the GPU you want to use (e.g., index 0)
#     tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
# import tensorflow as tf

# # Enable eager execution
# tf.config.run_functions_eagerly(True)

def eval(model, data, l2_reg, batch_size, isrank, metric_scope, _print=False):
    preds = []
    losses = []

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, loss = model.eval(data_batch, l2_reg, 1)
        preds.extend(pred)
        losses.append(loss)

    loss = sum(losses) / len(losses)
    labels = data[4]
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))

    res = evaluate_multi(labels, preds, cate_ids, metric_scope, isrank, _print)

    print("EVAL TIME: %.4fs" % (time.time() - t))
    # return loss, res_low, res_high
    return loss, res


def func(data):
    click = np.array(data[4])
    click = np.minimum(1, np.sum(click, axis=1))
    return click

def create_ave_reward(reward, seq_len):
    return [[np.sum(reward[i])/seq_len[i] if j < seq_len[i] else 0
             for j in range(len(reward[0]))] for i in range(len(reward))]

def eval_pv_evaluator(model, data, l2_reg, batch_size, isrank, metric_scope, _print=False):
    preds = []
    # labels = []
    losses = []
    res = [[] for i in range(5)]  # [5, 4]

    data_size = len(data[0])
    batch_num = data_size // batch_size
    clicks = []
    print('eval', batch_size, batch_num)

    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, loss, b_s = model.eval(data_batch, l2_reg)
        preds.append(pred)
        losses.append(loss)
        clicks.append(func(data_batch))

    loss = sum(losses) / len(losses)
    for i in range(5):
        res[i] = np.array([-loss for i in range(4)])

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return loss, res


def eval_controllable(model, data, l2_reg, batch_size, isrank, metric_scope, _print=False, num_points = 30):
    preds = [[] for i in range(num_points)]
    losses = [[] for i in range(num_points)]

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    labels = data[4]
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))
    # prefs= np.linspace(2/3, 1.0, 20)
    prefs= np.linspace(0.0, 1.0, num_points)
    for i, pref in enumerate(prefs):
        for batch_no in range(batch_num):
            data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
            pred, loss = model.eval(data_batch, l2_reg, pref)
            preds[i].extend(pred)
            losses[i].append(loss)

    loss = [sum(loss) / len(loss) for loss in losses]  # [11]

    res = [[] for i in range(5)]  # [5, 11, 4]
    for pred in preds:
        r = evaluate_multi(labels, pred, cate_ids, metric_scope, isrank, _print)
        for j in range(5):
            res[j].append(r[j])

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return loss, res, prefs

def save_log_eval(file_path, new_data_to_append):
    # Đọc dữ liệu hiện tại từ file JSON
    # if os.path.exists(file_path):
    #     with open(file_path, "r") as json_file:
    #         existing_data = json.load(json_file)
    # # Thêm dữ liệu mới vào dữ liệu hiện tại
    #     existing_data.update(new_data_to_append)
    # # Ghi dữ liệu mới (bao gồm dữ liệu hiện tại và dữ liệu mới) vào file JSON
    #     with open(file_path, "w") as json_file:
    #         json.dump(existing_data, json_file, indent=2)
    #     print('Saved eval log')
    # else:
    with open(file_path, "w") as json_file:
        json.dump(new_data_to_append, json_file, indent=2)
        print('Saved eval log')




def train(train_file, test_file, feature_size, max_time_len, itm_spar_fnum, itm_dens_fnum, profile_num, params):
    tf.compat.v1.reset_default_graph()

    # gpu settings
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

    perlist = False
    if params.model_type == 'CMR_evaluator':
        model = CMR_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                              itm_dens_fnum,
                              profile_num, max_norm=params.max_norm)
    elif params.model_type == 'CMR_generator':
        model = CMR_generator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                              itm_dens_fnum,
                              profile_num, max_norm=params.max_norm, rep_num=params.rep_num,
                              acc_prefer=params.acc_prefer,
                              is_controllable=params.controllable)
        evaluator = CMR_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                                  itm_dens_fnum,
                                  profile_num, max_norm=params.max_norm)
        with evaluator.graph.as_default() as g:
            sess = tf.compat.v1.Session(graph=g, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
            evaluator.set_sess(sess)
            sess.run(tf.compat.v1.global_variables_initializer())
            evaluator.load(params.evaluator_path)
    else:
        print('No Such Model', params.model_type)
        exit(0)

    with model.graph.as_default() as g:
        sess = tf.compat.v1.Session(graph=g, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        model.set_sess(sess)
        model.load('/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/model/single_obj/10/202312181429_lambdaMART_CMR_generator_64_0.001_1e-05_64_16_0.8_single_model')
        

    training_monitor = {
        'train_loss': [],
        'auc_train_loss': [],
        'div_train_loss': [],
        'train_prefer': [],
        'vali_loss': [],
        'map_l': [],
        'ndcg_l': [],
        'clicks_l': [],
        'ilad_l': [],
        'err_ia_l': [],
    }

    training_monitor_2 = {
        'train_loss': [],
        'auc_train_loss': [],
        'div_train_loss': [],
        'train_prefer': [],
        'vali_loss': [],
        'map_l': [],
        'ndcg_l': [],
        'clicks_l': [],
        'ilad_l': [],
        'err_ia_l': [],
    }

    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(params.timestamp, initial_ranker, params.model_type,
                                                        params.batch_size,
                                                        params.lr, params.l2_reg, params.hidden_size, params.eb_dim,
                                                        params.keep_prob,
                                                        "controllable_LS_LSTM_adapt_trainEval" if params.controllable else 'single_model')
    if not os.path.exists('{}/logs_{}/{}'.format(parse.save_dir, data_set_name, max_time_len)):
        os.makedirs('{}/logs_{}/{}'.format(parse.save_dir, data_set_name, max_time_len))
    # if not os.path.exists('{}/save_model_{}/{}/{}/'.format(parse.save_dir, data_set_name, max_time_len, model_name)):
    #     os.makedirs('{}/save_model_{}/{}/{}/'.format(parse.save_dir, data_set_name, max_time_len, model_name))
    if not os.path.exists('{}/adaptation/{}/{}/'.format(parse.save_dir, max_time_len, model_name)):
        os.makedirs('{}/adaptation/{}/{}/'.format(parse.save_dir, max_time_len, model_name))
    # save_path = '{}/save_model_{}/{}/{}/ckpt'.format(parse.save_dir, data_set_name, max_time_len, model_name)
    save_path = '{}/adaptation/{}/{}/ckpt'.format(parse.save_dir, max_time_len, model_name)
    log_save_path = '{}/logs_{}/{}/{}.metrics'.format(parse.save_dir, data_set_name, max_time_len, model_name)
    
    model_name_eval = '{}_{}_{}_{}_{}_{}_{}_{}'.format(params.timestamp, initial_ranker, 'evaluator',
                                                        params.batch_size,
                                                        params.hidden_size, params.eb_dim,
                                                        params.keep_prob,
                                                        "controllable_LS_LSTM_adapt_trainEval" if params.controllable else 'single_model')
    if not os.path.exists('{}/evaluator/adaptation/{}/{}/'.format(parse.save_dir, max_time_len, model_name)):
        os.makedirs('{}/evaluator/adaptation/{}/{}/'.format(parse.save_dir, max_time_len, model_name))
    save_eval_path = '{}/evaluator/adaptation/{}/{}/ckpt'.format(parse.save_dir, max_time_len, model_name_eval)

    if not os.path.exists('checkpoints_pareto/generator/{}/'.format(model_name)):
        os.makedirs('checkpoints_pareto/generator/{}/'.format(model_name))
    if not os.path.exists('checkpoints_pareto/evaluator/{}/'.format(model_name_eval)):
        os.makedirs('checkpoints_pareto/evaluator/{}/'.format(model_name_eval))

    save_checkpoint_gen_path = 'checkpoints_pareto/generator/{}/ckpt'.format(model_name)
    save_checkpoint_eval_path = 'checkpoints_pareto/evaluator/{}/ckpt'.format(model_name_eval)
    file_log = '/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/log_eval/{}.json'.format('LS_LSTM_adapt_trainEval')

    train_losses_step = []
    auc_train_losses_step = []
    div_train_losses_step = []
    train_prefer_step = []

    # before training process
    step = 0

    # if not params.controllable:
    #     vali_loss, res = eval(model, test_file, params.l2_reg, params.batch_size, False, params.metric_scope)
    #     training_monitor['train_loss'].append(None)
    #     training_monitor['vali_loss'].append(None)
    #     training_monitor['map_l'].append(res[0][0])
    #     training_monitor['ndcg_l'].append(res[1][0])
    #     # training_monitor['de_ndcg_l'].append(res[2][0])
    #     training_monitor['clicks_l'].append(res[2][0])
    #     training_monitor['ilad_l'].append(res[3][0])
    #     training_monitor['err_ia_l'].append(res[4][0])
    #     training_monitor['alpha_ndcg'].append(res[5][0])
    #     # training_monitor['utility_l'].append(res[4][0])
    #     if params.with_evaluator_metrics:
    #         training_monitor['eva_sum'].append(res[-2][0])
    #         training_monitor['eva_ave'].append(res[-1][0])

    #     training_monitor_2['train_loss'].append(None)
    #     training_monitor_2['vali_loss'].append(None)
    #     training_monitor_2['map_l'].append(res[0].tolist())
    #     training_monitor_2['ndcg_l'].append(res[1].tolist())
    #     # training_monitor['de_ndcg_l'].append(res[2][0])
    #     training_monitor_2['clicks_l'].append(res[2].tolist())
    #     training_monitor_2['ilad_l'].append(res[3].tolist())
    #     training_monitor_2['err_ia_l'].append(res[4].tolist())
    #     training_monitor_2['alpha_ndcg'].append(res[5].tolist())
    #     # training_monitor['utility_l'].append(res[4][0])

    #     print("STEP %d  INTIAL RANKER | LOSS VALI: NULL" % step)
    #     # if not params.with_evaluator_metrics:
    #     for i, s in enumerate(params.metric_scope):
    #         print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ILAD: %.4f  ERR_IA: %.4f  ALPHA_NDCG: %.4f" % (
    #             s, res[0][i], res[1][i], res[2][i], res[3][i], res[4][i], res[5][i]))
    # else:
    #     vali_loss, res, prefs = eval_controllable(model, test_file, params.l2_reg, params.batch_size, False,
    #                                        params.metric_scope)
    #     training_monitor['train_loss'].append(None)
    #     training_monitor['vali_loss'].append(None)
    #     training_monitor['map_l'].append(res[0][-1][-1])
    #     training_monitor['ndcg_l'].append(res[1][-1][-1])
    #     # training_monitor['de_ndcg_l'].append(res[2][0])
    #     training_monitor['clicks_l'].append(res[2][-1][-1])
    #     # training_monitor['utility_l'].append(res[4][0])
    #     training_monitor['ilad_l'].append(res[3][0][-2])
    #     training_monitor['err_ia_l'].append(res[4][0][-1])

    #     training_monitor_2['train_loss'].append(None)
    #     training_monitor_2['vali_loss'].append(None)
    #     training_monitor_2['map_l'].append(res[0])
    #     training_monitor_2['ndcg_l'].append(res[1])
    #     # training_monitor['de_ndcg_l'].append(res[2][0])
    #     training_monitor_2['clicks_l'].append(res[2])
    #     training_monitor_2['ilad_l'].append(res[3])
    #     training_monitor_2['err_ia_l'].append(res[4])
    #     # for j in [0, 5, 10]:

    #     MAP, NDCG, ILAD, ERR_IA = [],[],[],[]
    #     for j in range(10):
    #         print("auc_prefer: ", prefs[j])
    #         print("STEP %d  INTIAL RANKER | LOSS VALI: NULL" % step)
    #         for i, s in enumerate(params.metric_scope):
    #             print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ILAD: %.4f  ERR_IA: %.4f" % (
    #                 s, res[0][j][i], res[1][j][i], res[2][j][i], res[3][j][i], res[4][j][i]))
    #     res = np.array(res)
        
        # for i, k in enumerate(params.metric_scope):
        #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/initRank/initRank_eval_MAP_NDCG@{k}.npy',np.vstack((np.array(res[0,:,i]), np.array(res[1,:,i]))))
        #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/initRank/initRank_eval_MAP_ILAD@{k}.npy',np.vstack((np.array(res[0,:,i]), np.array(res[3,:,i]))))
        #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/initRank/initRank_eval_MAP_ERR_IA@{k}.npy',np.vstack((np.array(res[0,:,i]), np.array(res[4,:,i]))))
        #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/initRank/initRank_eval_NDCG_ILAD@{k}.npy',np.vstack((np.array(res[1,:,i]), np.array(res[3,:,i]))))
        #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/initRank/initRank_eval_NDCG_ERR_IA@{k}.npy',np.vstack((np.array(res[1,:,i]), np.array(res[4,:,i]))))


    early_stop = False

    data = train_file
    data_size = len(data[0])
    batch_num = data_size // params.batch_size
    # eval_iter_num = (data_size) // params.batch_size
    eval_iter_num = batch_num
    print('train', data_size, batch_num)

    max_pareto_ratio = 0.0
    log_eval = dict()
    print('begin training process')
    for epoch in range(params.epoch_num):
        # if early_stop:
        #     break
        # list_pref = [0, 0.2, 0.4, 0.6, 0.8]
        for batch_no in range(batch_num):
            data_batch = get_aggregated_batch(data, batch_size=params.batch_size, batch_no=batch_no)
            # if early_stop:
            #     break
            # for pref in list_pref:
            # acc_pref_ = random.uniform(0.8, 1)
            # div_pref_ = random.uniform(0.0, 0.4)
            train_prefer = random.uniform(0.0, 1.0)
            # train_prefer = 1
            # train_prefer = acc_pref_ / (abs(acc_pref_) + abs(div_pref_))

        # train_prefer = random.vonmisesvariate(0.9,500)
        # ray = [np.cos(train_prefer),np.sin(train_prefer)]
        # ray /= sum(ray)
        # ray *= np.random.randint(1, 5)*abs(np.random.normal(1, 0.2)) #some trick to make hypernetwork is more general
            if step % 3000 ==0:
                lr_decay_factor =  step//3000 + 1
            train_prefer_step.append(train_prefer)
            if params.model_type == 'CMR_generator':
                training_attention_distribution, training_prediction_order, predictions, cate_seq, cate_chosen = \
                    model.rerank(data_batch, params.keep_prob, train_prefer=train_prefer)

                rl_sp_outputs, rl_de_outputs = model.build_ft_chosen(data_batch, training_prediction_order)
                rerank_click = np.array(model.build_label_reward(data_batch[4], training_prediction_order))
                auc_rewards = evaluator.predict(np.array(data_batch[1]), rl_sp_outputs, rl_de_outputs, data_batch[6])
                base_auc_rewards = evaluator.predict(np.array(data_batch[1]), np.array(data_batch[2]),
                                                    np.array(data_batch[3]), data_batch[6])

                auc_rewards -= base_auc_rewards

                _, base_div_rewards = model.build_erria_reward(cate_seq, cate_seq)   # rank base rerank new
                _, div_rewards = model.build_erria_reward(cate_chosen, cate_seq)
                div_rewards -= base_div_rewards

                loss,auc_loss, div_loss = model.train(data_batch, training_prediction_order, auc_rewards, div_rewards,
                                                    params.lr, params.l2_reg, params.keep_prob,
                                                    train_prefer=train_prefer)
                # if epoch == 0:
                loss_evaluator  = evaluator.train(data_batch, 5e-4, 2e-4, params.keep_prob,
                                                    train_prefer=train_prefer )
                
                auc_train_losses_step.append(auc_loss)
                div_train_losses_step.append(div_loss)
                wandb.log({'train_loss/loss_evaluator':loss_evaluator,'train_loss/total_loss': loss,'train_loss/auc':auc_train_losses_step[-1],'train_loss/div':div_train_losses_step[-1]})
            else:
                loss = model.train(data_batch, params.lr, params.l2_reg, params.keep_prob, train_prefer)
            step += 1
            train_losses_step.append(loss)
            # model.train_epoch()

            if step % eval_iter_num == 0:
                train_loss = sum(train_losses_step) / len(train_losses_step)
                training_monitor['train_loss'].append(train_loss)
                train_losses_step = []
                ave_train_prefer = sum(train_prefer_step) / len(train_prefer_step)
                training_monitor['train_prefer'].append(ave_train_prefer)
                train_prefer_step = []
                auc_train_loss = sum(auc_train_losses_step) / len(auc_train_losses_step) if len(
                    auc_train_losses_step) else 0
                training_monitor['auc_train_loss'].append(auc_train_loss)
                auc_train_losses_step = []
                div_train_loss = sum(div_train_losses_step) / len(div_train_losses_step) if len(
                    div_train_losses_step) else 0
                training_monitor['div_train_loss'].append(div_train_loss)
                div_train_losses_step = []

                if not params.controllable:
                    print('Eval: single loss')
                    vali_loss, res = eval(model, test_file, params.l2_reg, 2018, True,
                                          params.metric_scope, False)
                    wandb.log({'train_loss':train_loss,'auc_train_loss':auc_train_loss,'div_train_loss':div_train_loss,'vali_loss': vali_loss})
                    training_monitor['train_loss'].append(train_loss)
                    training_monitor['train_prefer'].append(params.acc_prefer)
                    training_monitor['auc_train_loss'].append(auc_train_loss)
                    training_monitor['div_train_loss'].append(div_train_loss)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['map_l'].append(res[0][0])
                    training_monitor['ndcg_l'].append(res[1][0])
                    # training_monitor['de_ndcg_l'].append(res[2][0])
                    training_monitor['clicks_l'].append(res[2][0])
                    # training_monitor['utility_l'].append(res[4][0])
                    training_monitor['ilad_l'].append(res[3][0])
                    training_monitor['err_ia_l'].append(res[4][0])

                    training_monitor_2['train_loss'].append(train_loss)
                    training_monitor_2['vali_loss'].append(vali_loss)
                    training_monitor_2['map_l'].append(res[0].tolist())
                    training_monitor_2['ndcg_l'].append(res[1].tolist())
                    # training_monitor['de_ndcg_l'].append(res[2][0])
                    training_monitor_2['clicks_l'].append(res[2].tolist())
                    training_monitor_2['ilad_l'].append(res[3].tolist())
                    training_monitor_2['err_ia_l'].append(res[4].tolist())
                    # training_monitor['utility_l'].append(res[4][0])

                    print("EPOCH %d STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f" % (epoch, step, train_loss, vali_loss))
                    print("TRAIN PREFER: %.4f | AUC LOSS TRAIN: %.4f | DIV LOSS TRAIN: %.4f" % (
                        params.acc_prefer, auc_train_loss, div_train_loss))
                    for i, s in enumerate(params.metric_scope):
                        print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ILAD: %.4f  ERR_IA: %.4f" % (
                            s, res[0][i], res[1][i], res[2][i], res[3][i], res[4][i]))
                        
                    # res = np.array(res)
                    # for i, k in enumerate(params.metric_scope):
                    #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/single_loss/{epoch}_{step}_eval_MAP_NDCG@{k}.npy',np.vstack((np.array(res[0,i]), np.array(res[1,i]))))
                    #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/single_loss/{epoch}_{step}_eval_MAP_ILAD@{k}.npy',np.vstack((np.array(res[0,i]), np.array(res[3,i]))))
                    #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/single_loss/{epoch}_{step}_eval_MAP_ERR_IA@{k}.npy',np.vstack((np.array(res[0,i]), np.array(res[4,i]))))
                    #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/single_loss/{epoch}_{step}_eval_NDCG_ILAD@{k}.npy',np.vstack((np.array(res[1,i]), np.array(res[3,i]))))
                    #     np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/single_loss/{epoch}_{step}_eval_NDCG_ERR_IA@{k}.npy',np.vstack((np.array(res[1,i]), np.array(res[4,i]))))
                    wandb.log({'MAP/map@5_1':res[0][0],
                        'MAP/map@10_1': res[0][1],
                        'NDCG/ndcg@5_1':res[1][0],
                        'NDCG/ndcg@10_1':res[1][1], 
                        'ILAD/ilad@5_1':res[3][0],
                        'ILAD/ilad@10_1':res[3][1], 
                        'ERR_IA/err_ia@5_1':res[4][0],
                        'ERR_IA/err_ia@10_1':res[4][1]})

                    if training_monitor['map_l'][-1] >= max(training_monitor['map_l'][:]):
                        # save model
                        model.save(save_path)
                        evaluator.save(save_eval_path)
                        pkl.dump(res[-1], open(log_save_path, 'wb'))
                        print('model saved')

                    if len(training_monitor['map_l']) > 2 and epoch > 0:
                        # if (training_monitor['vali_loss'][-1] > training_monitor['vali_loss'][-2] and
                        #         training_monitor['vali_loss'][-2] > training_monitor['vali_loss'][-3]):
                        #     early_stop = True
                        if (training_monitor['map_l'][-2] - training_monitor['map_l'][-1]) <= 0.01 and (
                                training_monitor['map_l'][-3] - training_monitor['map_l'][-2]) <= 0.01:
                            early_stop = True
                else:
                    # pass
                    vali_loss, res, prefs = eval_controllable(model, test_file, params.l2_reg, 4096, True,
                                                       params.metric_scope, False)
                    training_monitor['train_loss'].append(train_loss)
                    training_monitor['train_prefer'].append(ave_train_prefer)
                    training_monitor['auc_train_loss'].append(auc_train_loss)
                    training_monitor['div_train_loss'].append(div_train_loss)
                    training_monitor['vali_loss'].append(vali_loss)

                    #log loss
                    wandb.log({'train_loss':train_loss,'auc_train_loss':auc_train_loss,'div_train_loss':div_train_loss,'vali_loss': vali_loss})
                    training_monitor['map_l'].append(res[0][-1][-1])
                    training_monitor['ndcg_l'].append(res[1][-1][-1])
                    # training_monitor['de_ndcg_l'].append(res[2][0])
                    training_monitor['clicks_l'].append(res[2][-1][-1])
                    # training_monitor['utility_l'].append(res[4][0])
                    training_monitor['ilad_l'].append(res[3][0][-2])
                    training_monitor['err_ia_l'].append(res[4][0][-1])

                    training_monitor_2['train_loss'].append(train_loss)
                    training_monitor_2['vali_loss'].append(vali_loss)
                    training_monitor_2['map_l'].append(res[0])
                    training_monitor_2['ndcg_l'].append(res[1])
                    # training_monitor['de_ndcg_l'].append(res[2][0])
                    training_monitor_2['clicks_l'].append(res[2])
                    training_monitor_2['ilad_l'].append(res[3])
                    training_monitor_2['err_ia_l'].append(res[4])

                    print("EPOCH %d STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f" % (
                        epoch, step, train_loss, sum(vali_loss) / len(vali_loss)))
                    print("TRAIN PREFER: %.4f | AUC LOSS TRAIN: %.4f | DIV LOSS TRAIN: %.4f" % (
                        ave_train_prefer, auc_train_loss, div_train_loss))
                    # for j in [0, 5, 10]:
                    for j in [0, 14, 29]:
                        print("auc_prefer: ", prefs[j])
                        print("STEP %d  INTIAL RANKER | LOSS VALI: NULL" % step)
                        for i, s in enumerate(params.metric_scope):
                            print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ILAD: %.4f  ERR_IA: %.4f" % (
                                s, res[0][j][i], res[1][j][i], res[2][j][i], res[3][j][i], res[4][j][i]))
                    res = np.array(res)
                    # wandb.log({'New_lambda/MAP/map@5_0':res[0][0][0], 'New_lambda/MAP/map@5_0.5':res[0][4][0], 'New_lambda/MAP/map@5_1':res[0][9][0],
                    #            'New_lambda/MAP/map@10_0': res[0][0][1], 'New_lambda/MAP/map@10_0.5': res[0][4][1], 'New_lambda/MAP/map@10_1':res[0][9][1],
                    #            'New_lambda/NDCG/ndcg@5_0':res[1][0][0], 'New_lambda/NDCG/ndcg@5_0.5':res[1][4][0], 'New_lambda/NDCG/ndcg@5_1':res[1][9][0],
                    #            'New_lambda/NDCG/ndcg@10_0':res[1][0][1], 'New_lambda/NDCG/ndcg@10_0.5':res[1][4][1], 'New_lambda/NDCG/ndcg@10_1':res[1][9][1],
                    #            'New_lambda/ILAD/ilad@5_0':res[3][0][0],'New_lambda/ILAD/ilad@5_0.5':res[3][4][0], 'New_lambda/ILAD/ilad@5_1':res[3][9][0],
                    #            'New_lambda/ILAD/ilad@10_0':res[3][0][1], 'New_lambda/ILAD/ilad@10_0.5':res[3][4][1], 'New_lambda/ILAD/ilad@10_1':res[3][9][1],
                    #            'New_lambda/ERR_IA/err_ia@5_0':res[4][0][0], 'New_lambda/ERR_IA/err_ia@5_0.5':res[4][4][0], 'New_lambda/ERR_IA/err_ia@5_1':res[4][9][0],
                    #            'New_lambda/ERR_IA/err_ia@10_0':res[4][0][1], 'New_lambda/ERR_IA/err_ia@10_0.5}':res[4][4][1], 'New_lambda/ERR_IA/err_ia@10_1':res[4][9][1]})

                
                    
                    for i, k in enumerate(params.metric_scope):
                        np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/adaptation/LS_LSTM_adapt_trainEval/{epoch}_{step}_eval_MAP_NDCG@{k}.npy',np.vstack((np.array(res[0,:,i]), np.array(res[1,:,i]))))
                        np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/adaptation/LS_LSTM_adapt_trainEval/{epoch}_{step}_eval_MAP_ILAD@{k}.npy',np.vstack((np.array(res[0,:,i]), np.array(res[3,:,i]))))
                        np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/adaptation/LS_LSTM_adapt_trainEval/{epoch}_{step}_eval_MAP_ERR_IA@{k}.npy',np.vstack((np.array(res[0,:,i]), np.array(res[4,:,i]))))
                        np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/adaptation/LS_LSTM_adapt_trainEval/{epoch}_{step}_eval_NDCG_ILAD@{k}.npy',np.vstack((np.array(res[1,:,i]), np.array(res[3,:,i]))))
                        np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/adaptation/LS_LSTM_adapt_trainEval/{epoch}_{step}_eval_NDCG_ERR_IA@{k}.npy',np.vstack((np.array(res[1,:,i]), np.array(res[4,:,i]))))
                    
                    data_eval = np.vstack((np.array(res[0,:,0]), np.array(res[4,:,0])))
                    data_eval = data_eval.T
                    pr = Metric().pareto_ratio(data_eval)
                    if pr > max_pareto_ratio:
                        max_pareto_ratio = pr
                        model.save(save_checkpoint_gen_path)
                        evaluator.save(save_checkpoint_eval_path)
                        print('save checkpoint pareto')
                        if epoch not in log_eval:
                            log_eval[epoch] = dict()
                        log_eval[epoch]['pr'] = pr
                        sp = Metric().spacing(data_eval)
                        hv = Metric().calculate_hypervolume(data_eval)
                        hrs = Metric().Hole_relative_size(data_eval)
                        log_eval[epoch]['hrs'] = hrs
                        log_eval[epoch]['hv'] = hv
                        log_eval[epoch]['sp'] = sp
                        save_log_eval(file_log,log_eval)

                    wandb.log({'MAP/map@5_0':res[0][0][0], 'MAP/map@5_0.5':res[0][14][0], 'MAP/map@5_1':res[0][29][0],
                        'MAP/map@10_0': res[0][0][1], 'MAP/map@10_0.5': res[0][14][1], 'MAP/map@10_1':res[0][29][1],
                        'NDCG/ndcg@5_0':res[1][0][0], 'NDCG/ndcg@5_0.5':res[1][14][0], 'NDCG/ndcg@5_1':res[1][29][0],
                        'NDCG/ndcg@10_0':res[1][0][1], 'NDCG/ndcg@10_0.5':res[1][14][1], 'NDCG/ndcg@10_1':res[1][29][1],
                        'ILAD/ilad@5_0':res[3][0][0],'ILAD/ilad@5_0.5':res[3][14][0], 'ILAD/ilad@5_1':res[3][29][0],
                        'ILAD/ilad@10_0':res[3][0][1], 'ILAD/ilad@10_0.5':res[3][14][1], 'ILAD/ilad@10_1':res[3][29][1],
                        'ERR_IA/err_ia@5_0':res[4][0][0], 'ERR_IA/err_ia@5_0.5':res[4][14][0], 'ERR_IA/err_ia@5_1':res[4][29][0],
                        'ERR_IA/err_ia@10_0':res[4][0][1], 'ERR_IA/err_ia@10_0.5}':res[4][14][1], 'ERR_IA/err_ia@10_1':res[4][29][1]})
                    if training_monitor['map_l'][-1] >= max(training_monitor['map_l'][:]):
                        # save model
                        model.save(save_path)
                        pkl.dump(res[-1], open(log_save_path, 'wb'))
                        print('model saved')

                        # if epoch == 0:
                        # evaluator.save(save_eval_path)
                        # print('eval saved')
                    

            # generate log
            if not os.path.exists('{}/logs_{}/{}/'.format(parse.save_dir, data_set_name, max_time_len)):
                os.makedirs('{}/logs_{}/{}/'.format(parse.save_dir, data_set_name, max_time_len))
            with open('{}/logs_{}/{}/{}.monitor.pkl'.format(parse.save_dir, data_set_name, max_time_len, model_name),
                      'wb') as f:
                pkl.dump(training_monitor, f)
            with open('{}/logs_{}/{}/{}.monitor2.pkl'.format(parse.save_dir, data_set_name, max_time_len, model_name),
                      'wb') as f:
                pkl.dump(training_monitor_2, f)

        # if epoch %5 ==0:
        #         vali_loss, res, prefs = eval_controllable(model, data, params.l2_reg,1024, True,
        #                                                params.metric_scope, False, num_points=30)
        #         res = np.array(res)
        #         for i, k in enumerate(params.metric_scope):
        #                 np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/train/new_hyper/chebyshev_LSTM/{epoch}_{step}_eval_MAP_NDCG@{k}.npy',np.vstack((np.array(res[0,:,i]), np.array(res[1,:,i]))))
        #                 np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/train/new_hyper/chebyshev_LSTM/{epoch}_{step}_eval_MAP_ILAD@{k}.npy',np.vstack((np.array(res[0,:,i]), np.array(res[3,:,i]))))
        #                 np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/train/new_hyper/chebyshev_LSTM/{epoch}_{step}_eval_MAP_ERR_IA@{k}.npy',np.vstack((np.array(res[0,:,i]), np.array(res[4,:,i]))))
        #                 np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/train/new_hyper/chebyshev_LSTM/{epoch}_{step}_eval_NDCG_ILAD@{k}.npy',np.vstack((np.array(res[1,:,i]), np.array(res[3,:,i]))))
        #                 np.save(f'/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/result/train/new_hyper/chebyshev_LSTM/{epoch}_{step}_eval_NDCG_ERR_IA@{k}.npy',np.vstack((np.array(res[1,:,i]), np.array(res[4,:,i]))))
        #         wandb.log({'train/MAP/map@5_0':res[0][0][0], 'train/MAP/map@5_0.5':res[0][14][0], 'train/MAP/map@5_1':res[0][29][0],
        #                 'train/MAP/map@10_0': res[0][0][1], 'train/MAP/map@10_0.5': res[0][14][1], 'train/MAP/map@10_1':res[0][29][1],
        #                 'train/NDCG/ndcg@5_0':res[1][0][0], 'train/NDCG/ndcg@5_0.5':res[1][14][0], 'train/NDCG/ndcg@5_1':res[1][29][0],
        #                 'train/NDCG/ndcg@10_0':res[1][0][1], 'train/NDCG/ndcg@10_0.5':res[1][14][1], 'train/NDCG/ndcg@10_1':res[1][29][1],
        #                 'train/ILAD/ilad@5_0':res[3][0][0],'train/ILAD/ilad@5_0.5':res[3][14][0], 'train/ILAD/ilad@5_1':res[3][29][0],
        #                 'train/ILAD/ilad@10_0':res[3][0][1], 'train/ILAD/ilad@10_0.5':res[3][14][1], 'train/ILAD/ilad@10_1':res[3][29][1],
        #                 'train/ERR_IA/err_ia@5_0':res[4][0][0], 'train/ERR_IA/err_ia@5_0.5':res[4][14][0], 'train/ERR_IA/err_ia@5_1':res[4][29][0],
        #                 'train/ERR_IA/err_ia@10_0':res[4][0][1], 'train/ERR_IA/err_ia@10_0.5}':res[4][14][1], 'train/ERR_IA/err_ia@10_1':res[4][29][1]})

        # if epoch % 5 == 0 and params.controllable:
        #     ctl_save_path = '{}/new_hyper/{}/{}/{}/ckpt'.format(parse.save_dir, max_time_len,
        #                                                             model_name,
        #                                                             epoch)
        #     model.save(ctl_save_path)
        #     print('model saved')


def reranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time_len', default=10, type=int, help='max time length')
    parser.add_argument('--save_dir', type=str, default='./', help='dir that saves logs and model')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/Data/ad/', help='data dir')
    parser.add_argument('--model_type', default='CMR_generator',
                        choices=['CMR_generator', 'CMR_evaluator'],
                        type=str,
                        help='algorithm name')
    parser.add_argument('--data_set_name', default='ad', type=str, help='name of dataset, including ad and prm')
    parser.add_argument('--initial_ranker', default='lambdaMART', choices=['DNN', 'lambdaMART'], type=str,
                        help='name of dataset, including DNN, lambdaMART')
    parser.add_argument('--epoch_num', default=30, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--rep_num', default=5, type=int, help='samples repeat number')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--l2_reg', default=1e-4, type=float, help='l2 loss scale')
    parser.add_argument('--keep_prob', default=0.8, type=float, help='keep probability')
    parser.add_argument('--eb_dim', default=16, type=int, help='size of embedding')
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size')
    parser.add_argument('--group_size', default=1, type=int, help='group size for GSF')
    parser.add_argument('--acc_prefer', default=1.0, type=float, help='accuracy_prefer/(accuracy_prefer+diversity)')
    parser.add_argument('--metric_scope', default=[5, 10], type=list, help='the scope of metrics')
    parser.add_argument('--max_norm', default=0, type=float, help='max norm of gradient')
    parser.add_argument('--c_entropy', default=0.001, type=float, help='entropy coefficient in loss')
    # parser.add_argument('--decay_steps', default=3000, type=int, help='learning rate decay steps')
    # parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--evaluator_path', type=str, default='', help='evaluator ckpt dir')
    parser.add_argument('--reload_path', type=str, default='', help='model ckpt dir')
    parser.add_argument('--setting_path', type=str, default='./example/config/ad/cmr_generator_setting.json',
                        help='setting dir')
    parser.add_argument('--controllable', type=bool, default=True, help='is controllable')

    # multisample
    parser.add_argument('--n_mo_sol', type = int, default= 16)
    parser.add_argument('--n_mo_obj', type=int, default=2)
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    set_global_determinism(1234)
    parse = reranker_parse_args()
    if parse.setting_path:
        parse = load_parse_from_json(parse, parse.setting_path)

    wandb.init(
        project='MORS-reranking',
        name= 'LS_LSTM_adapt_trainEval',
        config={
  "model_type": "CMR_generator",
  "data_set_name": "ad",
  "initial_ranker": "lambdaMART",
  "lr": parse.lr,
  "l2_reg": parse.l2_reg,
  "batch_size": parse.batch_size,
  "max_time_len": 10,
  "metric_scope": [5, 10],
  "eb_dim": parse.eb_dim,
  "hidden_size": parse.hidden_size,
  "epoch_num": parse.epoch_num,
  "controllable": True,
  "acc_prefer": [0,1],
  "new":'emb prefer 32, new_LSTM ',
  'save_folder': 'Controllable-Multi-Objective-Reranking-v2/result/adaptation/LS_LSTM_adapt_trainEval'
}
    )

    data_set_name = parse.data_set_name
    processed_dir = parse.data_dir
    stat_dir = os.path.join(processed_dir, 'data.stat')
    max_time_len = parse.max_time_len
    initial_ranker = parse.initial_ranker
    if data_set_name == 'prm' and parse.max_time_len > 30:
        max_time_len = 30
    print(parse)

    with open(stat_dir, 'r') as f:
        stat = json.load(f)

    num_item, num_cate, num_ft, profile_fnum, itm_spar_fnum, itm_dens_fnum, = stat['item_num'], stat['cate_num'], \
                                                                              stat['ft_num'], stat['profile_fnum'], \
                                                                              stat['itm_spar_fnum'], stat[
                                                                                  'itm_dens_fnum']
    print('num of item', num_item, 'num of list', stat['train_num'] + stat['val_num'] + stat['test_num'],
          'profile num', profile_fnum, 'spar num', itm_spar_fnum, 'dens num', itm_dens_fnum)
    # train_file, val_file, test_file = pkl.load(open(os.path.join(processed_dir, 'data.data'), 'rb'))
    # props = pkl.load(open(os.path.join(processed_dir, 'prop'), 'rb'))
    # props[0] = [1e-6 for i in range(max_time_len)]
    # profile = pkl.load(open(os.path.join(processed_dir, 'user.profile'), 'rb'))

    # construct training files
    train_dir = os.path.join(processed_dir, initial_ranker + '.data.train')

    if os.path.isfile(train_dir):
        train_lists = pkl.load(open(train_dir, 'rb'))
    else:
        train_lists = construct_list(os.path.join(processed_dir, initial_ranker + '.rankings.train'), max_time_len)
        pkl.dump(train_lists, open(train_dir, 'wb'))

    # construct test files
    test_dir = os.path.join(processed_dir, initial_ranker + '.data.test')
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        test_lists = construct_list(os.path.join(processed_dir, initial_ranker + '.rankings.test'), max_time_len)
        pkl.dump(test_lists, open(test_dir, 'wb'))

    train(train_lists, test_lists, num_ft, max_time_len, itm_spar_fnum, itm_dens_fnum, profile_fnum, parse)
    # model = CMR_generator(num_ft, parse.eb_dim, parse.hidden_size, max_time_len, itm_spar_fnum,
    #                           itm_dens_fnum,
    #                           profile_fnum, max_norm=parse.max_norm, rep_num=parse.rep_num,
    #                           acc_prefer=parse.acc_prefer,
    #                           is_controllable=parse.controllable)
    # model.load('/home/ubuntu/duc.nm195858/Controllable-Multi-Objective-Reranking-v2/model/save_model_ad/10/202312100044_lambdaMART_CMR_generator_16_0.0001_1e-05_64_16_0.8_controllable/ckpt')
    # print(model.summary())
    wandb.finish()
