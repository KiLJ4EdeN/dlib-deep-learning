// Dlib transfer-learning implementation, from dlib/examples
#include <dlib/dnn.h>
#include <iostream>

// This header file includes a generic definition of the most common ResNet architectures
#include "resnet.h"

using namespace std;
using namespace dlib;


// use the image input, res50 backbone then a 128 fc
namespace model
{
    template<template<typename> class BN>
    using net_type = loss_metric<
        fc_no_bias<128,
        avg_pool_everything<
        typename resnet::def<BN>::template backbone_50<
        input_rgb_image
        >>>>;

    using train = net_type<bn_con>;
    using infer = net_type<affine>;
}

// Custom weight decay modification class
class visitor_weight_decay_multiplier
{
public:

    visitor_weight_decay_multiplier(double new_weight_decay_multiplier_) :
        new_weight_decay_multiplier(new_weight_decay_multiplier_) {}

    template <typename layer>
    void operator()(layer& l) const
    {
        set_weight_decay_multiplier(l, new_weight_decay_multiplier);
    }

private:

    double new_weight_decay_multiplier;
};


int main() try
{
    // set model to train mode
    model::train net;

    // scope creation
    {
        // Now, let's define the classic ResNet50 network and load the pretrained model on
        // ImageNet.
        resnet::train_50 resnet50;
        std::vector<string> labels;
        deserialize("resnet50_1000_imagenet_classifier.dnn") >> resnet50 >> labels;

        // extract the backbone
        auto backbone = std::move(resnet50.subnet().subnet());

        // get the resnet backbone
        net.subnet().subnet() = backbone;

        // stack new layers on top
        using net_type = loss_metric<fc_no_bias<128, decltype(backbone)>>;
        net_type net2;
        net2.subnet().subnet() = backbone;
    }

    // apply the decay multiplier
    visit_computational_layers(net, visitor_weight_decay_multiplier(0.001));

    // set lr for all layers (example)
    set_all_learning_rate_multipliers(net, 0.5);

    // freezes every layer
    visit_computational_layers(net.subnet().subnet(), visitor_weight_decay_multiplier(0));
    set_all_learning_rate_multipliers(net.subnet().subnet(), 0);

    // set rate for a range of layers
    visit_computational_layers_range<0, 2>(net, visitor_weight_decay_multiplier(1));

    // differnet lr across the network layers
    set_learning_rate_multipliers_range<  0,   2>(net, 1);
    set_learning_rate_multipliers_range<  2,  38>(net, 0.1);
    set_learning_rate_multipliers_range< 38, 107>(net, 0.01);
    set_learning_rate_multipliers_range<107, 154>(net, 0.001);
    set_learning_rate_multipliers_range<154, 193>(net, 0.0001);

    // create a dummy image
    matrix<rgb_pixel> image(224, 224);
    assign_all_pixels(image, rgb_pixel(0, 0, 0));
    // create a mini batch to do prediction for 1 image
    std::vector<matrix<rgb_pixel>> minibatch(1, image);
    resizable_tensor input;
    net.to_tensor(minibatch.begin(), minibatch.end(), input);
    net.forward(input);
    cout << net << endl;
    // get the input details
    cout << "input size=(" <<
       "num:" << input.num_samples() << ", " <<
       "k:" << input.k() << ", " <<
       "nr:" << input.nr() << ", "
       "nc:" << input.nc() << ")" << endl;

    // get number of parameters
    cout << "number of network parameters: " << count_parameters(net) << endl;

    // any fine tuning can be done here

    return EXIT_SUCCESS;
}
catch (const serialization_error& e)
{
    cout << e.what() << endl;
    cout << "You need to download a copy of the file resnet50_1000_imagenet_classifier.dnn" << endl;
    cout << "available at http://dlib.net/files/resnet50_1000_imagenet_classifier.dnn.bz2" << endl;
    cout << endl;
    return EXIT_FAILURE;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}

