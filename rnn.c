#include <stddef.h>

void cpu_rnn_wipe(RNN *n){
    for(int i = 0; i < n->depth; i++){
        RNN_layer *l = &n->layers[i];
        for(int j = 0; j < l->size; j++)
            l->loutput[j] = 0.0f;
    }
}
RNN_layer cpu_create_RNN_layer(size_t input_dim, size_t size, float *params, int param_offset, Nonlinearity logistic){
    RNN_layer l;
    l.size = size;
    l.logistic = logistic;
    l.param_offset = param_offset;
    l.input_dimension = input_dim + size;

    size_t neuron_offset = param_offset;
    for(int i = 0; i < size; i++){
        xavier_init(&params[neuron_offset], (l.input_dimension+1), l.size);
        neuron_offset += l.input_dimension+1;
    }

    l.z = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.output = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.input_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.input_dimension);

    l.input   = ALLOC(float *, SIEKNET_MAX_UNROLL_LENGTH);
    l.loutput = ALLOC(float, l.size);

    cpu_zero_2d_arr(l.input_gradient, SIEKNET_MAX_UNROLL_LENGTH, l.input_dimension);
    return l;
}

RNN cpu_rnn_from_arr(const size_t *arr, const size_t len) {
    RNN n;
    n.t = 0;
    n.stateful = 0;
    n.seq_len = 25;
    n.input_dimension = arr[0];
    n.output_dimension = arr[len-1];
    n.depth = len-2;
    n.cost_fn = cross_entropy;

    if(len < 3){
        printf("ERROR: rnn_from_arr(): must have at least one input dim, hidden layer size, and output dim (3 layers) but only %lu layers provided.\n", len);
        exit(1);
    }

    size_t num_params = 0;
    for(int i = 1; i < len-1; i++){
        num_params += ((arr[i-1]+arr[i]+1)*arr[i]); //parameters for input, recurrent input, and bias term
    }
    num_params += (arr[len-2]+1)*arr[len-1]; //output mlp layer params
    n.num_params = num_params;

    n.layers = ALLOC(RNN_layer, len-2);
    n.params = ALLOC(float, num_params);
    n.param_grad = ALLOC(float, num_params);

    memset(n.param_grad, '\0', num_params*sizeof(float));

    int param_idx = 0;
    for(int i = 1; i < len-1; i++){
        RNN_layer l = cpu_create_RNN_layer(arr[i-1], arr[i], n.params, param_idx, hypertan);
        param_idx += (arr[i-1]+arr[i]+1)*arr[i];
        n.layers[i-1] = l;
    }
    n.output_layer = cpu_create_MLP_layer(arr[len-2], arr[len-1], n.params, param_idx, softmax);

    n.recurrent_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, arr[len-2]);
    n.network_input      = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, arr[0]);

    n.cost_gradient = ALLOC(float, n.output_dimension);

    cpu_zero_2d_arr(n.recurrent_gradient, SIEKNET_MAX_UNROLL_LENGTH, arr[len-2]);
    cpu_zero_2d_arr(n.network_input, SIEKNET_MAX_UNROLL_LENGTH, arr[0]);

    n.output = n.output_layer.output;
    return n;
}

void cpu_rnn_layer_forward(RNN_layer *l, float *x, const float *params, size_t t){
    l->input[t] = x;
    int params_per_neuron = l->input_dimension+1;
    for(int i = 0; i < l->size; i++){
        l->z[t][i] = 0;
        agnostic_rnn_forward_kernel(x, l->loutput, l->z[t], params, l->input_dimension, l->size, l->param_offset, params_per_neuron, i);
        l->output[t][i] = activate(l->z[t][i], l->logistic);
    }
    for(int i = 0; i < l->size; i++)
        l->loutput[i] = l->output[t][i];
}

static void cpu_lstm_wipe(LSTM *n){
    for(int i = 0; i < n->depth; i++){
        LSTM_layer *l = &n->layers[i];
        cpu_zero_2d_arr(&l->loutput, 1, l->size);
        cpu_zero_2d_arr(&l->lstate, 1, l->size);
    }
    cpu_zero_2d_arr(n->recurrent_gradient, SIEKNET_MAX_UNROLL_LENGTH, n->layers[n->depth-1].size);
    n->t = 0;
}

LSTM_layer cpu_create_LSTM_layer(size_t input_dim, size_t size, float *params, int param_offset){
    LSTM_layer l;
    l.size = size;
    l.param_offset = param_offset;
    l.input_dimension = input_dim + size;

    size_t neuron_offset = param_offset;
    for(int i = 0; i < size*4; i++){
        xavier_init(&params[neuron_offset], (l.input_dimension+1), l.size);
        neuron_offset += l.input_dimension+1;
    }

    l.input_nonl_z  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.input_gate_z  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.forget_gate_z = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.output_gate_z = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

    l.input_nonl_output  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.input_gate_output  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.forget_gate_output = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.output_gate_output = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

    l.input_nonl_gradient  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.input_gate_gradient  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.forget_gate_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.output_gate_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

    l.input_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.input_dimension);
    l.cell_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

    l.cell_state = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
    l.cell_dstate = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

    l.input = ALLOC(float*, SIEKNET_MAX_UNROLL_LENGTH);
    l.output = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

    l.loutput = ALLOC(float, l.size);
    l.lstate  = ALLOC(float, l.size);

    cpu_zero_2d_arr(l.input_gradient, SIEKNET_MAX_UNROLL_LENGTH, l.input_dimension);

    return l;
}

void cpu_lstm_forward(LSTM *n, float *x){
    size_t t = n->t;
    for(int i = 0; i < n->input_dimension; i++){
        n->network_input[t][i] = x[i];
    }
    //Feedforward through all layers
    float *input = n->network_input[t];
    for(int i = 0; i < n->depth; i++){
        LSTM_layer *l = &n->layers[i];
        cpu_lstm_layer_forward(l, input, n->params, t);
        input = l->output[t];
    }
    cpu_mlp_layer_forward(&n->output_layer, input, n->params);
}