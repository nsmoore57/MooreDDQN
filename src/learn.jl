function learn!(env::E, qpolicy::Q, mem::M, num_eps, γ;
                maxn=200, opt=ADAM(0.00001), update_freq=3000, chkpt_freq=3000,
                chkpt_filename="model_checkpoint.bson", cb_ep = () -> (),
                cb_step = () -> (), show_progress=true) where {E<:Reinforce.AbstractEnvironment,
                                                               Q<:QPolicy,
                                                               M<:ReplayMemoryBuffer}

    # Build an epsilon greedy policy for the learning
    π = ϵGreedyPolicy(LinearSequence(1.,0.01,num_eps*50,iter=0), qpolicy)

    # Track number of sucessful attempts
    num_successes = 0

    # Params to optimize
    p = get_params(qpolicy)

    losses = Float64[]

    # Track the number of training steps completed so far
    step = 1
    show_progress && (progress = Progress(num_eps, 3))
    for i ∈ 1:num_eps
        ep = Episode(env, π; maxn = maxn)

        for (s, a, r, s′) ∈ ep
            # Save the step into the replay buffer
            addexp!(mem, s(), a, r, s′(), finished(env, s′))

            # Fill the buffer before training or lowering ϵ
            if length(mem) < mem.batch_size
                cb_step()
                continue
            end

            # Training
            td_errs = batchTrain!(π, mem, qpolicy, γ, p, opt)

            # If needed, update the target Network
            update_freq > 0 && step % update_freq == 0 && update_target(qpolicy)
            update_freq > 0 && step % update_freq == 0 && push!(losses, Flux.huber_loss(td_errs))

            # If desired, save the network
            chkpt_freq > 0 && step % chkpt_freq == 0 && save_policy(qpolicy)

            # Record if this episode was successful
            finished(env, s′) && (num_successes += 1)

            step += 1

            # Run the step callback
            cb_step()

        end # end of episode

        # Change β
        update_β!(mem)

        # Run the episode callback
        cb_ep()

        # Update the progress
        show_progress && next!(progress)
    end
    chkpt_freq > 0 && save_policy(qpolicy, chkpt_filename)
    num_successes, losses
end

function batchTrain!(π, mem, qpolicy, γ, params, opt)
    # Decrease ϵ
    update_ϵ!(π)

    # Sample a batch from the replay buffer
    (s_batch, a_batch, r_batch, s′_batch, done_batch, ids, weights) = sample(mem)

    # Get the target values based on the next states
    target = get_target(qpolicy, γ, r_batch, done_batch, s′_batch)

    # May need this to help set the data type for loss - not sure
    loss = 0.0

    # Track the td errs
    td_errs = similar(target)

    # Track gradients while calculating the loss
    gs = Flux.gradient(params) do
        currentQ_SA = get_QValues(qpolicy, s_batch)[a_batch]
        td_errs = currentQ_SA .- target
        loss = Flux.huber_loss(td_errs.*weights)
    end

    # Train the network(s)
    Flux.Optimise.update!(opt, params, gs)

    # Update Priorities for selected memory elements
    update_priorities!(mem, ids, td_errs)

    # return the td_errs so we can save them for later
    return td_errs
end
