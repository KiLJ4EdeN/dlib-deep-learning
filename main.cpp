// Dlib resnet implementation, from dlib/examples
#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;


// we build blocks using alias templates


template <
    int N,
    template <typename> class BN,
    int stride,
    typename SUBNET
    >
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

// skip connections
template <
    template <int,template<typename>class,int,typename> class block,
    int N,
    template<typename>class BN,
    typename SUBNET
    >
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

// downsampling resnet bloack
template <
    template <int,template<typename>class,int,typename> class block,
    int N,
    template<typename>class BN,
    typename SUBNET
    >
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;



// affine layers instead of batch norm for testing
template <typename SUBNET> using res       = relu<residual<block,8,bn_con,SUBNET>>;
template <typename SUBNET> using ares      = relu<residual<block,8,affine,SUBNET>>;
template <typename SUBNET> using res_down  = relu<residual_down<block,8,bn_con,SUBNET>>;
template <typename SUBNET> using ares_down = relu<residual_down<block,8,affine,SUBNET>>;



// with the help of aliases the main network is defined
const unsigned long number_of_classes = 10;
using net_type = loss_multiclass_log<fc<number_of_classes,
                            avg_pool_everything<
                            res<res<res<res_down<
                            repeat<9,res, // repeat this layer 9 times
                            res_down<
                            res<
                            input<matrix<unsigned char>>
                            >>>>>>>>>>;


// prelu block
template <typename SUBNET>
using pres  = prelu<add_prev1<bn_con<con<8,3,3,1,1,prelu<bn_con<con<8,3,3,1,1,tag1<SUBNET>>>>>>>>;

// ----------------------------------------------------------------------------------------

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
    std::vector<unsigned long> training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long> testing_labels;
    // load the data using the provided function from dlib
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);


    // slower cuDNN methods for big data
    set_dnn_prefer_smallest_algorithms();


    // creating an initial network
    net_type net;
    // change the number of fc neurons in the last layer
    net_type net2(num_fc_outputs(15));

    // this shows a way to replace relu with prelu
    using net_type2 = loss_multiclass_log<fc<number_of_classes,
                                avg_pool_everything<
                                pres<res<res<res_down< // 2 prelu layers here
                                tag4<repeat<9,pres,    // 9 groups, each containing 2 prelu layers
                                res_down<
                                res<
                                input<matrix<unsigned char>>
                                >>>>>>>>>>>;
    net_type2 pnet(prelu_(0.2),
                   prelu_(0.25),
                   repeat_group(prelu_(0.3),prelu_(0.4))
                   );


    // model inspection
    cout << "The pnet has " << pnet.num_layers << " layers in it." << endl;
    // this is like model.summary()
    cout << pnet << endl;

    // get a layers output by index
    layer<3>(pnet).get_output();
    // or get params for vis purposes
    cout << "prelu param: "<< layer<7>(pnet).layer_details().get_initial_param_value() << endl;

    // get layers by their type
    layer<tag1>(pnet);
    layer<tag4,1>(pnet);
    layer<tag4,2>(pnet);


    // trainer with adam instead of sgd
    dnn_trainer<net_type,adam> trainer(net,adam(0.0005, 0.9, 0.999));
    // below code to parallelize training on gpu devices
    //dnn_trainer<net_type,adam> trainer(net,adam(0.0005, 0.9, 0.999), {0,1});

    trainer.be_verbose();
    trainer.set_max_num_epochs(1);
    // scheduler and earlystopping
    trainer.set_iterations_without_progress_threshold(2000);
    trainer.set_learning_rate_shrink_factor(0.1);
    // The learning rate will start at 1e-3.
    trainer.set_learning_rate(1e-3);
    trainer.set_synchronization_file("mnist_resnet_sync", std::chrono::seconds(100));


    // here we use steps to train on a hypothetical dataset that does not fit in ram
    std::vector<matrix<unsigned char>> mini_batch_samples;
    std::vector<unsigned long> mini_batch_labels;
    dlib::rand rnd(time(0));

    while(trainer.get_learning_rate() >= 1e-6)
    {
        mini_batch_samples.clear();
        mini_batch_labels.clear();

        // creating the batch with 128 size
        while(mini_batch_samples.size() < 128)
        {
            auto idx = rnd.get_random_32bit_number()%training_images.size();
            mini_batch_samples.push_back(training_images[idx]);
            mini_batch_labels.push_back(training_labels[idx]);
        }

        // step once with the selected batch
        trainer.train_one_step(mini_batch_samples, mini_batch_labels);

        // also can use trainer.test_one_step for validation data
    }

    // get net, when parallelizing training, waits for all subprocesses to end and gets the model
    trainer.get_net();


    net.clean();
    serialize("mnist_res_network.dat") << net;


    // replace batch norm layers
    using test_net_type = loss_multiclass_log<fc<number_of_classes,
                                avg_pool_everything<
                                ares<ares<ares<ares_down<
                                repeat<9,ares,
                                ares_down<
                                ares<
                                input<matrix<unsigned char>>
                                >>>>>>>>>>;
    // create a testing net and load the model again
    test_net_type tnet = net;
    deserialize("mnist_res_network.dat") >> tnet;


    // predict using the model and see how many of the images were classified correctly

    std::vector<unsigned long> predicted_labels = tnet(training_images);
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < training_images.size(); ++i)
    {
        if (predicted_labels[i] == training_labels[i])
            ++num_right;
        else
            ++num_wrong;

    }
    cout << "training num_right: " << num_right << endl;
    cout << "training num_wrong: " << num_wrong << endl;
    cout << "training accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;

    predicted_labels = tnet(testing_images);
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
    cout << "testing accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}


