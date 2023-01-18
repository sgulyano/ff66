import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from math import prod

from model.dataset import split2feat

def multilabel_classification_report(dataset, model, n_features=None):
    X_test, y_test = dataset.get_test_data()
    if n_features:
        X_test = split2feat(X_test, n_features)
    pred = model.predict(X_test)
    
    res = model.evaluate(X_test, y_test)
    # print(f'Loss: {res[0]:.4f}, Binary Accuracy: {res[1]:.4f}')
    
    acc = np.logical_and(y_test == 1, (pred>=0.5)).sum(axis=0)# / y_test.sum(axis=0)
    # print(acc.sum(), y_test.sum(axis=0).sum())
    print(f"Accuracy:  {acc.sum()} / {y_test.sum(axis=0).sum()}")

    results = []
    for i in range(dataset.get_num_class()):
    #     print(pred[:,i].shape)
    #     print(y_test[:,i].shape)
        p, r, f1, sup = precision_recall_fscore_support(y_test[:,i].astype('int8'), (pred[:,i]>0.5).astype('int8'), average=None)
        results.append((p[1], r[1], f1[1], sup[1]))
    #     print(f"{c} {p[1]:.3f}, {r[1]:.3f}, {f1[1]:.3f}, {sup[1]}")

    print(f'{" ":10s}  precision     recall   f1-score    support')
    for n, (p ,r ,f1, sup) in zip(dataset.class_name, results):
        print(f'{n:10s}  {p:9.2f}  {r:9.2f}  {f1:9.2f}  {sup:9d}')
    p ,r ,f1, sup = np.array(results).mean(axis=0)
    print("")
    print(f'{"macro avg":10s}  {p:9.2f}  {r:9.2f}  {f1:9.2f}  {len(X_test):9d}')


def multilabel_classification_report_pred(y_test, y_pred, class_name):
    acc = np.logical_and(y_test == 1, (y_pred>=0.5)).sum(axis=0)# / y_test.sum(axis=0)
    print(f"Accuracy:  {acc.sum()} / {y_test.sum(axis=0).sum()}")
    s = f"#### Accuracy:  {acc.sum()} / {y_test.sum(axis=0).sum()}  \n"

    results = []
    for i in range(len(class_name)):
        p, r, f1, sup = precision_recall_fscore_support(y_test[:,i].astype('int8'), (y_pred[:,i]>0.5).astype('int8'), average=None)
        results.append((p[1], r[1], f1[1], sup[1]))

    print(f'{" ":10s}  precision     recall   f1-score    support')
    s += "| | precision | recall | f1-score | support |  \n"
    s += "|---|---|---|---|---|  \n"
    for n, (p ,r ,f1, sup) in zip(class_name, results):
        print(f'{n:10s}  {p:9.2f}  {r:9.2f}  {f1:9.2f}  {sup:9d}')
        s += f"| {n:10s} | {p:9.2f} | {r:9.2f} | {f1:9.2f} | {sup:9d}  \n"
    p ,r ,f1, sup = np.array(results).mean(axis=0)
    print("")
    s += "|---|---|---|---|---|  \n"
    print(f'{"macro avg":10s}  {p:9.2f}  {r:9.2f}  {f1:9.2f}  {len(y_test):9d}')
    s += f'| {"macro avg":10s} | {p:9.2f} | {r:9.2f} | {f1:9.2f} | {len(y_test):9d}  \n'
    return s


def get_codebook_indices(vqvae, X):
    encoder = vqvae.get_layer("encoder")
    quantizer = vqvae.get_layer("vector_quantizer")

    # Generate the codebook indices.
    encoded_outputs = encoder.predict(X)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

    return codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])


def unknown_prediction_report(pred, class_name, fl):
    s_file = "| Filename | Predicted Types |  \n|---|---|  \n"
    s_type = "| Type | Count |  \n|---|---|  \n"
    pred_bin = pred>=0.5
    res = []
    for p in pred_bin:
        res.append(class_name[p])
    
    print('Prediction per file')
    for n, t in sorted(zip(fl, res)):
        if len(t) > 0:
            print(f"{os.path.basename(n):25s}, {t}")
            s_file += f"| {os.path.basename(n):25s} | {t} |  \n"
    
    print('No. of prediction per material type')
    for t, n in zip(class_name, pred_bin.sum(axis=0)):
        print(f"{t:10s}: {n}")
        s_type += f"| {t:10s} | {n} |  \n"
    # return pred, pred_bin, res
    return s_file, s_type


def plot_reconstruction(original, reconstructed):
    plt.figure(figsize=(10,2))
    plt.plot(original, label='Original')
    plt.plot(reconstructed.squeeze(), label='Reconstruction')
    plt.legend()
    plt.show()


def show_reconstruction(X, model):
    idx = np.random.choice(len(X), 10)
    test_images = X[idx]
    reconstructions_test = model.predict(test_images)

    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        plot_reconstruction(test_image, reconstructed_image)


def plot_reconstruction_codebook(X, reconstruct, codebook, s):
    # plt.figure(figsize=(12,2), gridspec_kw={"width_ratios" : b.shape})
    
    fig, axs = plt.subplots(1, 2, figsize=(12,2), gridspec_kw={"width_ratios" : (4,1)})
    
    axs[0].plot(X, label='Original')
    axs[0].plot(reconstruct, label='Reconstruction')
    axs[0].legend()
    axs[0].title.set_text("Reconstruction")

    # plt.subplot(1, 2, 2)
    cb = np.pad(codebook, ((0,prod(s)-codebook.size)), mode='constant').reshape(s)
    axs[1].imshow(cb)
    axs[1].title.set_text("Codebook")

def show_reconstruction_codebook(X, model, s):
    encoder = model.get_layer("encoder")
    quantizer = model.get_layer("vector_quantizer")
    decoder = model.get_layer("decoder")

    encoded_outputs = encoder.predict(X)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    X_codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    reconstruct = decoder.predict(encoded_outputs)
    
    print(f"Codebook size is {X_codebook_indices.shape[1]}")

    for i in range(10):
        plot_reconstruction_codebook(X[i], reconstruct[i], X_codebook_indices[i], s)

