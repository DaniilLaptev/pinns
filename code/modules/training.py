
import json
from datetime import datetime
from collections import defaultdict

def train(
    problem,
    model,
    hyperparameters,
    logging_params = None,
    plotting_params = None,
    name = None,
    print_every = None
) -> dict:
    """
    Function for training PINN model to solve specific differential equation.
    
    Parameters
    ----------
    problem : problem-like
        Should be either instance of class Problem or provide minimum functionality that
        will be required during training: domain as dictionary 
    
    """
    start_time = datetime.now()
    
    logging = {}
    
    if logging_params is None:
        ll = ln = le = ls = -1
        log_dir = None
    else:
        ll  = logging_params['loss']
        ln = logging_params['norms']
        le = logging_params['error']
        ls = logging_params['steps']
        log_dir = logging_params['log_dir']
        
    if log_dir is None:
        if name is None:
            log_dir = f'./logging/{start_time.strftime("%d.%m.%Y_%H.%M.%S")}.json'
        else:
            log_dir = f'./logging/{name}.json'
        
    optimizer = hyperparameters['optimizer'](model.parameters(), **hyperparameters['arguments'])
    lr_scheduler = hyperparameters['scheduler']['rule'](optimizer, **hyperparameters['scheduler']['parameters'])
    
    lbcoefs, lfcoefs = hyperparameters['coefs']
    dynamic_coefs = hyperparameters['dynamic_coefs']
    if dynamic_coefs > -1:
        coef_func = hyperparameters['coef_func']
    
    num_iters = hyperparameters['num_iters']
    if le == 0: le = num_iters - 1
        
    cp_params = hyperparameters['collocation_points']
    generation_function = cp_params['generation_function']
    N_points = cp_params['N_points']
    sample_every = cp_params['sample_every']
    if sample_every == 0:
        boundary_points, inner_points = generation_function(problem, **N_points, loss_history=[])
    
    if plotting_params is None:
        plot_every = -1
    else:
        plot_every = plotting_params['plot_every']
        save_dir = plotting_params['save_dir']
        size = plotting_params['size']
    
    params = model.get_params()
    
    loss_history = []
        
    for i in range(num_iters):
        current_log = defaultdict(None)
        
        error_calculated = False
        if le > -1 and i % le == 0:
            predictions = problem.test(model)
            error = problem.error(predictions)
            error_calculated = True
            current_log['error'] = error

        if sample_every > 0 and i % sample_every == 0:
            boundary_points, inner_points = generation_function(problem=problem, **N_points, loss_history=loss_history)
            
        if plot_every > -1 and i % plot_every == 0:
            if not error_calculated:
                predictions = problem.test(model)
                error = problem.error(predictions)
            problem.plot(predictions, name + f'_iteration_{i}', f'Iteration {i}', size, save_dir)
        
        optimizer.zero_grad()
        
        L_B = problem.boundary_loss(model, boundary_points, lbcoefs)
        L_F = problem.inner_loss(model, inner_points, lfcoefs)
        L = L_B + L_F
        L.backward()
        
        if ll > -1 and i % ll == 0:
            loss_history.append([float(L_B.item()), float(L_F.item()), float(L.item())])
            current_log['loss'] = loss_history[-1]
            
        if dynamic_coefs > -1 and i % dynamic_coefs == 0:
            lbcoefs, lfcoefs = coef_func(loss_history)
            
        if ln > -1 and i % ln == 0:
            current_log['norms'] = model.get_norms()
        
        optimizer.step()
        if ls > 0 and i % ls == 0:
            curr_params = model.get_params()
            current_log['step'] = float((curr_params - params).norm(p=2).item())
        
        logging[i] = current_log
        
        if i % print_every == 0:
            itstr = str(i)
            print(f'Iter {" "*(5 - len(itstr))+itstr} \t {error}')
            
        lr_scheduler.step()
    
    current_log = defaultdict(None)
    
    predictions = problem.test(model)
    error = problem.error(predictions)
    current_log['error'] = error
    
    logging[i+1] = current_log
        
    with open(log_dir, 'w') as log:
        json.dump(logging, log, indent=4)
        
    print(f'Final error: {error}')
    
    return {
        "final_error": error,
        "duration": (start_time - datetime.now()).seconds
    }