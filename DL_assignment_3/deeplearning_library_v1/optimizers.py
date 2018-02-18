import autograd.numpy as np
from autograd import value_and_grad
from autograd.misc.flatten import flatten_func

# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha_choice,max_its,w,version,beta_choice):
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g, w) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)
    
    if version == 'normalized':
        # run the gradient descent loop
        weight_history = []      # container for weight history
        cost_history = []        # container for corresponding cost function history
        alpha = alpha_choice
        beta = beta_choice
        z = np.zeros((np.shape(w)))
            
        for k in range(0,max_its):
            # evaluate the gradient, compute its length
            cost_eval,grad_eval = gradient(w)
            grad_norm = np.linalg.norm(grad_eval)

            weight_history.append(unflatten(w))
            cost_history.append(cost_eval)

            # check that magnitude of gradient is not too small, if yes pick a random direction to move
            if grad_norm == 0:
                # pick random direction and normalize to have unit legnth
                grad_eval = 10**-6*np.sign(2*np.random.rand(len(w)) - 1)
                grad_norm = np.linalg.norm(grad_eval)

            grad_eval /= grad_norm

            # take gradient descent step
            z = beta*z + grad_eval
            w = w - alpha*z

        # collect final weights
        weight_history.append(unflatten(w))
        # compute final cost function value via g itself (since we aren't computing 
        # the gradient at the final step we don't get the final cost function value 
        # via the Automatic Differentiatoor) 
        cost_history.append(g_flat(w))  

    elif version == 'unnormalized':
        # run the gradient descent loop
        weight_history = []      # container for weight history
        cost_history = []        # container for corresponding cost function history
        alpha = 0
        beta = beta_choice
        z = np.zeros((np.shape(w)))
        
        for k in range(0,max_its):
            # check if diminishing steplength rule used
            if alpha_choice == 'diminishing':
                alpha = 1/float(k)
            else:
                alpha = alpha_choice

            # evaluate the gradient, store current (unflattened) weights and cost function value
            cost_eval,grad_eval = gradient(w)
            weight_history.append(unflatten(w))
            cost_history.append(cost_eval)

            # take gradient descent step
            z = beta*z + grad_eval
            w = w - alpha*z

        # collect final weights
        weight_history.append(unflatten(w))
        # compute final cost function value via g itself (since we aren't computing 
        # the gradient at the final step we don't get the final cost function value 
        # via the Automatic Differentiatoor) 
        cost_history.append(g_flat(w))  
        
        
    return weight_history,cost_history