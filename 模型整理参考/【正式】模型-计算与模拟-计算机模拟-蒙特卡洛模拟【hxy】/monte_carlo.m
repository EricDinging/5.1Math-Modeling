function prob = monte_carlo(rand_times, gen, cond)
    cnt = 0;
    for i=1:rand_times
        x=gen();
        if cond(x)
            cnt = cnt + 1;
        end
    end
    prob = 1.0 * cnt / rand_times;
end
