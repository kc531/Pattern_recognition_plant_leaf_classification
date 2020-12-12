function [conf_matrix] = ConfusionMatrix(pred_labels, test_target, no_of_classes)
    conf_matrix = zeros(no_of_classes, no_of_classes);
    for i=1:no_of_classes
        for j=1:no_of_classes
            conf_matrix(i,j) = length(test_target(test_target==i & pred_labels==j));
        end
    end
end

