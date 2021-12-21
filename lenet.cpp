// Dlib lenet implementation, from dlib/examples
#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;

int main(int argc, char** argv) try
{
    // Parse path to mnist
    if (argc != 2)
    {
        cout << "This example needs the MNIST dataset to run!" << endl;
        cout << "You can get MNIST from http://yann.lecun.com/exdb/mnist/" << endl;
        cout << "Download the 4 files that comprise the dataset, decompress them, and" << endl;
        cout << "put them in a folder.  Then give that folder as input to this program." << endl;
        return 1;
    }

    // containers for ubyte mnist
    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;
    // load the data using the provided function from dlib
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);


    // net type definition
    using net_type = loss_multiclass_log<
                                fc<10,
                                relu<fc<84,
                                relu<fc<120,
                                max_pool<2,2,2,2,relu<con<16,5,5,1,1,
                                max_pool<2,2,2,2,relu<con<6,5,5,1,1,
                                input<matrix<unsigned char>>
                                >>>>>>>>>>>>;


    // use the net_type we just created to initiate an instance of net
    net_type net;
    // trainer with sgd, parameter setup
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(128);
    trainer.set_max_num_epochs(2);
    trainer.be_verbose();
    // this line does state saving based on seconds so we dont lose the results
    trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));
    // training start
    trainer.train(training_images, training_labels);

    // clean the net from any data and save it
    net.clean();
    serialize("mnist_network.dat") << net;
    // Now if we later wanted to recall the network from disk we can simply say:
    // deserialize("mnist_network.dat") >> net;


    // predict using the model and see how many of the images were classified correctly
    std::vector<unsigned long> predicted_labels = net(training_images);
    int num_right = 0;
    int num_wrong = 0;
    // And then let's see if it classified them correctly.
    for (size_t i = 0; i < training_images.size(); ++i)
    {
        if (predicted_labels[i] == training_labels[i])
            ++num_right;
        else
            ++num_wrong;

    }
    // calc accuracy
    cout << "training num_right: " << num_right << endl;
    cout << "training num_wrong: " << num_wrong << endl;
    cout << "training accuracy:  " << num_right / static_cast<double>(num_right + num_wrong) << endl;

    // test metrics
    predicted_labels = net(testing_images);
    num_right = 0;
    num_wrong = 0;
    for (size_t i = 0; i < testing_images.size(); ++i)
    {
        if (predicted_labels[i] == testing_labels[i])
            ++num_right;
        else
            ++num_wrong;

    }
    cout << "testing num_right: " << num_right << endl;
    cout << "testing num_wrong: " << num_wrong << endl;
    cout << "testing accuracy:  " << num_right / static_cast<double>(num_right + num_wrong) << endl;


    // save as xml, this can later be used to convert this to caffe
    net_to_xml(net, "net.xml");
}
// print the error exception if any happens
catch(std::exception& e)
{
    cout << e.what() << endl;
}

