import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def multilabel_classification_report(dataset, model):
    X_test, y_test = dataset.get_test_data()
    pred = model.predict(X_test)
    
    loss, categorical_acc = model.evaluate(X_test, y_test)
    print(f'Loss: {loss:.4f}, Binary Accuracy: {categorical_acc:.4f}')
    
    # acc = np.logical_and(y_test == 1, (pred>=0.5)).sum(axis=0) / y_test.sum(axis=0)

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