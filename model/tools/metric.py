import numpy as np


class SegmentationMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusionMatrix = np.zeros((self.num_classes, self.num_classes))

    def genConfusionMatrix(self, predict, label):
        """
        row: Ture label
        column: Predict result
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (label >= 0) & (label < self.num_classes)
        label = self.num_classes * label[mask] + predict[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusionMatrix = count.reshape(self.num_classes, self.num_classes)
        return confusionMatrix

    def pixelAccuracy(self):
        """
        return all class overall pixel accuracy
        PA = acc = (TP + TN) / (TP + TN + FP + TN)
        """
        Acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return Acc

    def classPixelAccuracy(self):
        """
        return each category pixel accuracy(A more accurate way to call it precision)
        Acc = (TP) / TP + FP
        返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
        """
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        """
        返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        """
        Intersection = TP
        Union = TP + FP + FN
        IoU = TP / (TP + FP + FN)
        """
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return IoU, mIoU

    def FrequencyWeightedIntersectionOverUnion(self):
        """
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, predict, label):
        assert predict.shape == label.shape
        self.confusionMatrix += self.genConfusionMatrix(predict, label)

    def reset(self):
        self.confusionMatrix = np.zeros((self.num_classes, self.num_classes))
