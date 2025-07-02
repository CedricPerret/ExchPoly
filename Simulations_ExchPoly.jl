#region Initialization

include(homedir()*"/OneDrive/Research/B1-Codes/Utility.jl")
include(homedir()*"/OneDrive/Research/B1-Codes/Toolbox_model2.jl")
using Plots
using LaTeXStrings
using StatsPlots
using Roots
using ForwardDiff
using Optim
using Base.Threads

"""
    fitness_function(population::Vector{Vector{Float64}}; o_x::Float64, o_y::Float64, sigma::Float64, alpha::Float64, exchange_fun::String, eta::Float64, args...)

This function computes the fitness of a population based on the production and exchange of two goods, `x` and `y`. The fitness value represents how well the population performs in terms of utility derived from the goods produced and exchanged.

# Arguments

- `population::Vector{Vector{Float64}}`: A vector where each element is a vector of size 2 containing `Float64` elements.
- `o_x::Float64`: Optimal production level for good `x`. This parameter determines the target production level that is considered optimal for good `x`.
- `o_y::Float64`: Optimal production level for good `y`. This parameter determines the target production level that is considered optimal for good `y`.
- `sigma::Float64`: production breadth parameter. It represents the flexibility or difficulty in adjusting production levels.
- `alpha::Float64`: Importance of good `x` for payoff.
- `exchange_fun::String`: Type of exchange mechanism used in the economy. It can be either `"No_exchange"`, where no exchange occurs, or `"walrasian_bargain"`, where a Walrasian bargaining process determines the exchange of goods.
- `eta::Float64`: Elasticity of scale parameter. Setting `eta` to 1 implies a fixed equal time allocation between the two goods.
- `args...`: Always required to use my personal evolutionary model

# Returns

- `w`: A vector or array representing the payoff of the population based on the goods produced and exchanged.
- `h`: A vector or array representing the time allocation to the production of good `x` for each individual or group in the population.
- `q_x`: A vector or array representing the quantity of good `x` produced by each individual or group in the population.
- `q_y`: A vector or array representing the quantity of good `y` produced by each individual or group in the population.
- `c_x`: A vector or array representing the amount of good `x` consumed after exchange for each individual or group in the population.
- `c_y`: A vector or array representing the amount of good `y` consumed after exchange for each individual or group in the population.

# Examples

population = [[0.5, 0.3], [0.6, 0.4], [0.7, 0.5]]  # Example population
o_x = -2.0
o_y = 2.0
sigma = 0.5
palpha = 0.7
exchange_fun = "walrasian_bargain"
eta = 0.95

w, h, q_x, q_y, c_x, c_y = fitness_function(population; o_x=o_x, o_y=o_y, sigma=sigma, alpha=palpha, exchange_fun=exchange_fun, eta=eta)

"""
function fitness_function(population; o_x,o_y,sigma,alpha,exchange_fun::String,eta,args...)
    #--- 1 Choose time allocation (see in Mathematica)
    if exchange_fun == "walrasian_bargain"
        #-> There are many pairs to process, so multithreading improves performance.
        h = threaded_find_h(population,eta,o_x,o_y,sigma, alpha)
    else
        h= find_h.(population,eta,o_x,o_y,sigma, alpha)
    end
    h_y = (x->1 .- x).(h)
    #--- 2 Produce initial quantities
    q_x = broadcast_nested(production_function,population,h,o_x,sigma,eta)
    q_y = broadcast_nested(production_function,population,h_y,o_y,sigma,eta)
    #--- 3 Exchange to obtain final quantities
    c = exchange_function.(exchange_fun,q_x,q_y,alpha)
    c_x = getindex.(c,1)
    c_y = getindex.(c,2)
    #--- 4 Calculate fitness using utility function
    w = broadcast_nested(utility_function,c_x,c_y,alpha)
    #--- Print additional output
    ## Variance in trait value
    if isa(o_x,Integer)
        #-> Two traits
        var_z1 = var(getindex.(vcat(population...),1);corrected=false)
        var_z2 = var(getindex.(vcat(population...),2);corrected=false)
        return(w,h,q_x,q_y,c_x,c_y,var_z1,var_z2)
    else
        #-> Single trait
        var_z = var(vcat(population...);corrected=false)
        return(w,h,q_x,q_y,c_x,c_y,var_z)
    end
end



#! Light version which outputs only the fitness and time allocation
function fitness_function_light(population; o_x,o_y,sigma,alpha,exchange_fun::String,eta,args...)
    fitness_function(population; o_x,o_y,sigma,alpha,exchange_fun,eta)[1:2]
end


"""
    find_h(group, eta, o_x, o_y, sigma, alpha)

Determines the time allocation.

# Arguments
- `group`: A tuple or vector of traits affecting productivity. If `length(group) > 2`, a market economy is assumed.
- `eta::Float64`: Controls the returns to scale:
  - `eta == 1`: Equal allocation.
  - `eta > 1`: Increasing returns to scale.
  - `eta < 1`: Allocation calculated numerically.
- `o_x::Float64`: Optimal production level for good `x`.
- `o_y::Float64`: Optimal production level for good `y`.
- `sigma::Float64`: production breadth parameter.
- `alpha::Float64`: Relative importance of good x

# Returns
- For `length(group) > 2`: A vector filled with `alpha` for market economies.
- Otherwise: A two-element vector `[h1, h2]`, representing time allocation to good `x`.

# Details
- For `eta == 1`, allocation is `[0.5, 0.5]`.
- For `eta > 1`:
  - If `tau == theta`, allocation is `[2, 2]` (resolved randomly to `[1, 0]` or `[0, 1]`).
  - Otherwise, the better trait for a task determines the allocation.
- For `eta < 1`, allocation is calculated by solving for a price (`p`) numerically using `fzero`.
- Random resolution is applied if two maximizers provide equal payoff.

# Example
```julia
find_h([0, 0.1], 0.7, -2, 2, 1, 0.5)
find_h([-2, 2], 0.7, -2, 2, 1, 0.5)
find_h([0.1, 0.2, 0.3, 0.4], 0.7, -2, 2, 1, 0.5)
find_h(rand(Uniform(-2,2),1000), 0.7, -2, 2, 1, 0.5)

"""
function find_h(group,eta,o_x,o_y,sigma, alpha)
    min = -2.5
    h= fill(0.,length(group))
    if eta == 0
        #-> version where the time allocation is equal and fixed
        h = fill(0.5,length(group))
    elseif eta > 1
        println("Increasing returns to scale not implemented. eta should be inferior to 1")
    else
        if typeof(o_x) != Int64
            error("The alternative production function is not implemented for VARYING time allocation")
        end
        #-> Diminishing returns to scale. We first determinate the price
        if length(group) == 2
            #-> Dyadic exchange
            #-> The price needs to be calculated numerically by solving the expression calculated in the mathematica notebook. Once price is found, the h are given by the formula found in the paper.
            f(p) = abs(symbolic_expression(productivity_function(group[1],o_x,sigma), productivity_function(group[1],o_y,sigma), productivity_function(group[2],o_x,sigma), productivity_function(group[2],o_y,sigma), eta, p));
            result = optimize(f, 1e-6, 1e10)  # Bound search to p > 0
            p = Optim.minimizer(result)

        else
            #-> Market economies
            p = try
               fzero((possible_price->does_market_clear(group,possible_price,o_x,o_y,eta,sigma,alpha)),1)
            catch e
                fzero((possible_price->does_market_clear(group,possible_price,o_x,o_y,eta,sigma,alpha)),BigFloat(1)) |> Float64
            end
          end
        h = h_expression.(p,group,o_x,o_y,eta,sigma)
  end
    return(h)
end


#@Much faster
function threaded_find_h(population, eta, o_x, o_y, sigma, alpha)
    h_population = Vector{Vector{Float64}}(undef, length(population)) 
    @threads for i in 1:length(population)
        h_population[i] = find_h(population[i], eta, o_x, o_y, sigma, alpha)
    end
    return h_population
end

function h_expression(p,z,o_x,o_y,eta,sigma)
  1 / (1  + ((1/p) *((productivity_function(z,o_y,sigma))./productivity_function(z,o_x,sigma))).^ (1/(1-eta)))
end

#@Using the equation for price. It takes sensibly the same time
function does_market_clear(population,p,o_x,o_y,eta,sigma,alpha)
    h=h_expression.(p,population,o_x,o_y,eta,sigma)
    sum_q_x= sum(production_function.(population,h,o_x,sigma,eta))
    sum_q_y= sum(production_function.(population,1 .- h,o_y,sigma,eta))
    (alpha/(1-alpha))*(sum_q_y/sum_q_x) -p
end

"""
    symbolic_expression(rxτ, ryτ, rxθ, ryθ, η, p)

This function represents the symbolic expression used to solve for diminishing returns to scale and dyadic exchange.

# Arguments
- `rxτ`, `ryτ`: Productivity parameters for trait `τ` with respect to goods `x` and `y`.
- `rxθ`, `ryθ`: Productivity parameters for trait `θ` with respect to goods `x` and `y`.
- `η`: Elasticity of scale parameter, influencing the returns to scale.
- `p`: The price parameter being solved for.

# Returns
The result of the symbolic expression combining productivity and allocation parameters.

# Example
f(p) = abs(symbolic_expression(1.,0.8,0.9,0.97,0.5,p));
result = optimize(f, 1e-6, 1e10)  # Bound search to p > 0
p = Optim.minimizer(result)
This yields 0.93159..., which is the same price than obtained in the mathematica notebook in the section Behavioural equilibrium
"""
function symbolic_expression(  rxτ, ryτ, rxθ,ryθ, η, p)
    term1 = 0.5 * ryθ * ((ryθ^(1 / (1 - η))) / ((p * rxθ)^(1 / (1 - η)) + ryθ^(1 / (1 - η))))^η
    term2 = -0.5 * (p * rxθ)^(1 / (1 - η)) * ((p * rxθ)^(1 / (1 - η)) + ryθ^(1 / (1 - η)))^(-η)
    term3 = 0.5 * ryτ * ((ryτ^(1 / (1 - η))) / ((p * rxτ)^(1 / (1 - η)) + ryτ^(1 / (1 - η))))^η
    term4 = -0.5 * (p * rxτ)^(1 / (1 - η)) * ((p * rxτ)^(1 / (1 - η)) + ryτ^(1 / (1 - η)))^(-η)
    return term1 + term2 + term3 + term4
end


"""
    production_function(z::Float64, h::Float64, optimal::Float64, sigma, eta::Float64)

Calculates the quantity `q` of a good produced based on the individual's trait `z`, time allocation `h`, optimal trait value `optimal`, production breadth `sigma`, and scale elasticity `eta`.
If optimal is an integer, then we use an alternative production function. Be careful, only eta 0 is implemented for this case.

# Arguments
- `z::Float64`: Trait of the individual.
- `h::Float64`: Time allocated to produce good.
- `optimal::Float64`: Optimal trait value for production.
- `sigma::Float64`: Sensitivity to deviations from optimal.
- `eta::Float64`: Elasticity of scale.

# Returns
- `q::Float64`: Quantity produced.
"""
function production_function(z,h,optimal::Float64,sigma,eta)
    q = h^eta * exp(-(z-optimal)^2/sigma^2)
end

function production_function(z,h,optimal::Integer,sigma,eta)
    q = h^eta * z[optimal]^sigma
end



"""
    production_function(z::Float64, h::Float64, optimal, sigma::Float64)

Calculates the productivity `r` of a good produced based on the individual's trait `z`,  optimal trait value `optimal` and production breadth `sigma`.
If optimal is an integer, then we use an alternative production function. Be careful, only eta 0 is implemented for this case.

# Arguments
- `z::Float64`: Trait of the individual.
- `optimal::Float64`: Optimal trait value for production.
- `sigma::Float64`: Sensitivity to deviations from optimal.

# Returns
- `r::Float64`: Quantity produced.
"""
function productivity_function(z,optimal::Float64,sigma)
    r = exp(-(z-optimal)^2/sigma^2)
end

function productivity_function(z,optimal::Integer,sigma)
    z[optimal]^sigma
end



"""
    exchange_function(type, q_x, q_y, alpha)

Computes the final consumption quantities of two goods (`x` and `y`) after a possible exchange between individuals.

# Arguments
- `type`: A `String` indicating the type of exchange. Can be:
  - `"no_exchange"`: No trade occurs; individuals consume what they produce.
  - `"walrasian_bargain"` or `"market"`: Exchange occurs according to a Walrasian equilibrium.
- `q_x`, `q_y`: Vectors of quantities of goods `x` and `y` produced by each individual.
- `alpha`: Preference parameter (between 0 and 1) determining the weight placed on good `x`.

# Returns
A tuple `(c_x, c_y)` containing the vectors of final consumption of goods `x` and `y` for each individual.

"""
function exchange_function(type,q_x,q_y,alpha)
    if type == "no_exchange"
        #-> Exchange do not take place, the final quantities = quantities produced by the individual
        #@ We reassign for clarity. Returning directly does not save much time (roughly 2ns)
        c_x,c_y = q_x,q_y
    elseif type == "walrasian_bargain" || type == "market"
        #-> The price and quantities exchanged are described by a Walrasian equilibrium between interacting individuals
        p = alpha / (1-alpha) * sum(q_y)/sum(q_x)
        #! The quantities of good x produced can be so low (e.g. eta = 0.95, sigma = 1) that the price is considered equal to 0 (floating point error) which leads to c_x = NaN. To deal with that, we correct these price to very low value of price
        if p == 0; p = 0.00000000000001 end
        c_x = alpha .* (q_x  .+ (1/p) .* q_y )
        c_y = (1-alpha) .* (p .* q_x .+ q_y)
        #@Sanity check. At Walrasian equilibre, the sum of excess demand function is equal to 0
        #println(sum(c_x .- q_x))
    end
    return(c_x,c_y)
end

"""
    utility_function(cx, cy, alpha)

Computes the Cobb-Douglas utility given consumption of two goods.

# Arguments
- `cx`: Consumption of good `x`.
- `cy`: Consumption of good `y`.
- `alpha`: Preference parameter (between 0 and 1) representing the weight given to good `x` in utility.

# Returns
A scalar representing the utility level derived from consuming `cx` units of good `x` and `cy` units of good `y`, using a Cobb-Douglas utility function:
"""
function utility_function(cx,cy,alpha)
    cx^alpha * cy^(1-alpha)
end

#region SANDBOX
    
parameters = Dict{Any,Any}(pairs((z_ini = Normal(0.,0.05), n_gen = 1000, n_ini = 2, n_patch = 500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "g",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(1), alpha = 0.5, eta = 0.95, other_output_name=["h","q_x","q_y","c_x","c_y"],
write_file=false, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print"],
j_print = 100,n_print=1,simplify=false)))

res=evol_model(reproduction_WF,fitness_function,parameters)

@df res plot(:gen,:mean_mean_q_x)

#endregion

cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul")

#region AUTARKY

#*** Autarky
cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh")

## Low production breadth sigma

parameters = Dict{Any,Any}(pairs((z_ini = 2., n_gen = 5000, n_ini = 2, n_patch = 2500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "no_exchange",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = 1, alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 200)))

res=evol_model(reproduction_WF,fitness_function,parameters)

## Low production breadth sigma and high alpha
parameters = Dict{Any,Any}(pairs((z_ini = 0., n_gen = 5000, n_ini = 2, n_patch = 500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "no_exchange",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = 1, alpha = 0.75, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 100)))

res=evol_model(reproduction_WF,fitness_function,parameters)

## Low production breadth sigma and low alpha
parameters = Dict{Any,Any}(pairs((z_ini = 0., n_gen = 5000, n_ini = 2, n_patch = 500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "no_exchange",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = 1, alpha = 0.25, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 100)))

res=evol_model(reproduction_WF,fitness_function,parameters)

## High production breadth sigma
#! We run it for longer as it takes more time to converge with high sigma
parameters = Dict{Any,Any}(pairs((z_ini = 2., n_gen = 10000, n_ini = 2, n_patch = 500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "no_exchange",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(5), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 200)))

res=evol_model(reproduction_WF,fitness_function,parameters)

## Very high production breadth sigma (for time allocation)
# We run it for longer as it takes more time to converge with high sigma
parameters = Dict{Any,Any}(pairs((z_ini = 2., n_gen = 15000, n_ini = 2, n_patch = 500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "no_exchange",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(10), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 300)))

res=evol_model(reproduction_WF,fitness_function,parameters)
#endregion

#region No time allocation => Eta 0
#*** Dyadic exchange
#--- Low production breadth sigma
##Load population from the end of the simulations without exchange
data=CSV.read(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh/"*"Exchpoly-alpha=0.5-de=i-eta=0-exchange_fun=no_exchange-n_gen=5.0k-n_patch=2.5k-n_simul=1.0-o_x=-2.0-o_y=2.0-sigma=1.0.csv",DataFrame,header=true,select=["gen", "z"] );
population_initial_low_sigma = [collect(x) for x in partition(dropdims(data.z[data.gen .== maximum(data.gen), :],dims=2),2)]
data = nothing

parameters = Dict{Any,Any}(pairs((z_ini = population_initial_low_sigma, n_gen = 7500, n_ini = 2, n_patch = 2500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = 1, alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 100)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#--- Low production breadth sigma and high alpha
#!Bigger pop
##Load population from the end of the simulations without exchange
data=CSV.read(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh/"*"Exchpoly-alpha=0.75-de=i-eta=0-exchange_fun=no_exchange-n_gen=5.0k-n_patch=2.5k-n_simul=1.0-o_x=-2.0-o_y=2.0-sigma=1.0.csv",DataFrame,header=true,select=["gen", "z"] );
population_initial_low_sigma = [collect(x) for x in partition(dropdims(data.z[data.gen .== maximum(data.gen), :],dims=2),2)]
data = nothing

parameters = Dict{Any,Any}(pairs((z_ini = population_initial_low_sigma, n_gen = 30000, n_ini = 2, n_patch = 2500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,n_simul=1,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = 1, alpha = 0.75, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 500)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#--- Low production breadth sigma and low alpha
#!Bigger pop
##Load population from the end of the simulations without exchange
data=CSV.read(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh/"*"Exchpoly-alpha=0.25-de=i-eta=0-exchange_fun=no_exchange-n_gen=5.0k-n_patch=2.5k-n_simul=1.0-o_x=-2.0-o_y=2.0-sigma=1.0.csv",DataFrame,header=true,select=["gen", "z"] );
population_initial_low_sigma = [collect(x) for x in partition(dropdims(data.z[data.gen .== maximum(data.gen), :],dims=2),2)]
data = nothing

parameters = Dict{Any,Any}(pairs((z_ini = population_initial_low_sigma, n_gen = 30000, n_ini = 2, n_patch = 2500,
boundaries=[-2.5,2.5],n_simul=1,
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = 1, alpha = 0.25, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 500)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#--- High production breadth sigma
##Load population from the end of the simulations without exchange
data=CSV.read(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh/"*"Exchpoly-alpha=0.5-de=i-eta=0-exchange_fun=no_exchange-n_gen=10.0k-n_patch=500-n_simul=1.0-o_x=-2.0-o_y=2.0-sigma=2.23607.csv",DataFrame,header=true,select=["gen", "z"] );
population_initial_high_sigma = [collect(x) for x in partition(dropdims(data.z[data.gen .== maximum(data.gen), :],dims=2),2)]
data = nothing

parameters = Dict{Any,Any}(pairs((z_ini = population_initial_high_sigma, n_gen = 20000, n_ini = 2, n_patch = 500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(5), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 200)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#*** Market

#--- High production breadth sigma
##Load population from the end of the simulations without exchange
data=CSV.read(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh/"*"Exchpoly-alpha=0.5-de=i-eta=0-exchange_fun=no_exchange-n_gen=10.0k-n_patch=500-n_simul=1.0-o_x=-2.0-o_y=2.0-sigma=2.23607.csv",DataFrame,header=true,select=["gen", "z"] );
population_initial_high_sigma_market = [dropdims(data.z[data.gen .== maximum(data.gen), :],dims=2)]
data = nothing

parameters = Dict{Any,Any}(pairs((z_ini = population_initial_high_sigma_market, n_gen = 20000, n_ini = 1000, n_patch = 1,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "market",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(5), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 200,simplify=false)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#endregion

#region TIME ALLOCATION DECISION - DIMINISHING returns to scale

cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/strath")

#*** Dyadic exchange

#--- High elasticity of scale eta. Econ -> Gen diversity
data=CSV.read(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh/"*"Exchpoly-alpha=0.5-de=i-eta=0-exchange_fun=no_exchange-n_gen=5.0k-n_patch=500-n_simul=1.0-o_x=-2.0-o_y=2.0-sigma=1.0.csv",DataFrame,header=true,select=["gen", "z"] );
population_initial = [collect(x) for x in partition(dropdims(data.z[data.gen .== maximum(data.gen), :],dims=2),2)]
data = nothing

parameters = Dict{Any,Any}(pairs((z_ini = population_initial, n_gen = 10000, n_ini = 2, n_patch = 500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = 1, alpha = 0.5, eta = 0.9, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 50)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#--- Low elasticity of scale eta + Low production breadth sigma. Gen diversity -> Econ
data=CSV.read(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh/"*"Exchpoly-alpha=0.5-de=i-eta=0-exchange_fun=no_exchange-n_gen=5.0k-n_patch=500-n_simul=1.0-o_x=-2.0-o_y=2.0-sigma=1.0.csv",DataFrame,header=true,select=["gen", "z"] );
population_initial_high_sigma = [collect(x) for x in partition(dropdims(data.z[data.gen .== maximum(data.gen), :],dims=2),2)]
data = nothing

parameters = Dict{Any,Any}(pairs((z_ini = population_initial_high_sigma, n_gen = 10000, n_ini = 2, n_patch = 500,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = 1, alpha = 0.5, eta = 0.5, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 50)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#--- Low elasticity of scale eta + High production breadth sigma
##Load population from the end of the simulations without exchange
cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/strath")
data=CSV.read(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh/"*"Exchpoly-alpha=0.5-de=i-eta=0-exchange_fun=no_exchange-n_gen=10.0k-n_patch=500-n_simul=1.0-o_x=-2.0-o_y=2.0-sigma=2.23607.csv",DataFrame,header=true,select=["gen", "z"] );
population_initial_high_sigma_dyadic = [dropdims(data.z[data.gen .== maximum(data.gen), :],dims=2)]
data = nothing

parameters = Dict{Any,Any}(pairs((z_ini = population_initial_high_sigma_dyadic, n_gen = 30000, n_ini = 2, n_patch = 500,n_simul=1,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(10), alpha = 0.5, eta = 0.5, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 150,simplify=false)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#--- Parameter sweep
cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/sweep")

parameters = Dict{Any,Any}(pairs((z_ini = Normal(0.,0.05), n_gen = 10000, n_ini = 2, n_patch = 500,n_simul=1,
boundaries= [-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(1), alpha = 0.5, eta = 0.5, other_output_name=["h"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 9999, n_print=1)))

@time evol_model(reproduction_WF,fitness_function_light,parameters)

sigmasq_list = collect(1:20)
eta_list = collect(0.6:0.1:0.9)

for eta_i in eta_list
    for sigmasq_i in sigmasq_list
        println("eta =",eta_i)
        println("sigma = ",sigmasq_i)
        parameters = Dict{Any,Any}(pairs((z_ini = Normal(0.,0.05), n_gen = 10000, n_ini = 2, n_patch = 500,n_simul=10,
        boundaries= [-2.5,2.5],
        mu_m = 0.01, sigma_m = 0.02,
        de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
        o_x = -2., o_y = 2., sigma = sqrt(sigmasq_i), alpha = 0.5, eta = eta_i, other_output_name=["h"],
        write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
        j_print = 9999, n_print=1)))
        evol_model(reproduction_WF,fitness_function_light,parameters)
    end
end


#*** Market
cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/strath")

#--- High production breadth sigma
##Load population from the end of the simulations without exchange
data=CSV.read(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh/"*"Exchpoly-alpha=0.5-de=i-eta=0-exchange_fun=no_exchange-n_gen=10.0k-n_patch=500-n_simul=1.0-o_x=-2.0-o_y=2.0-sigma=2.23607.csv",DataFrame,header=true,select=["gen", "z"] );
population_initial_high_sigma_market = [dropdims(data.z[data.gen .== maximum(data.gen), :],dims=2)]
data = nothing

parameters = Dict{Any,Any}(pairs((z_ini = population_initial_high_sigma_market, n_gen = 30000, n_ini = 1000, n_patch = 1,
boundaries=[-2.5,2.5],
mu_m = 0.01, sigma_m = 0.02,
de = "i",  exchange_fun = "market",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(10), alpha = 0.5, eta = 0.5, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
j_print = 150,simplify=false)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#--- Parameter sweep
cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/sweep")

sigmasq_list = collect(1:20)
eta_list = collect(0.05:0.05:0.95)

@time for eta_i in eta_list
    for sigmasq_i in sigmasq_list
        println("eta =",eta_i)
        println("sigmasq = ",sigmasq_i)
        parameters = Dict{Any,Any}(pairs((z_ini = Normal(0.,0.05), n_gen = 10000, n_ini = 1000, n_patch = 1,n_simul=10,
        boundaries= [-2.5,2.5],
        mu_m = 0.01, sigma_m = 0.02,
        de = "i",  exchange_fun = "market",name_model = "Exchpoly",
        o_x = -2., o_y = 2., sigma = sqrt(sigmasq_i), alpha = 0.5, eta = eta_i, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
        write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify"],
        j_print = 9999, n_print=1,simplify=false)))
        evol_model(reproduction_WF,fitness_function,parameters)
    end
end


#endregion

#region Robustness checks (with fixed time allocation )

#*** Baseline (for measure of variance)
cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh")

#--- Dyadic exchange

parameters = Dict{Any,Any}(pairs((z_ini = 2., n_gen = 10000, n_ini = 2, n_patch = 5000,n_simul=10,
boundaries=[-10.,10.],
mu_m = 0.01, sigma_m = 0.02, n_loci = 0,str_selection=1.,
de = "g",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(1), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","n_print","simplify"],
j_print = 250,n_print=1)))

res=evol_model(reproduction_WF,fitness_function,parameters)

#--- Market exchange

parameters = Dict{Any,Any}(pairs((z_ini = 2., n_gen = 10000, n_ini = 10000, n_patch = 1,n_simul=10,
boundaries=[-10.,10.],
mu_m = 0.01, sigma_m = 0.02, n_loci = 0,str_selection=1.,
de = "g",  exchange_fun = "market",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(1), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","n_print","simplify"],
j_print = 250,n_print=1,simplify=false)))

res=evol_model(reproduction_WF,fitness_function,parameters)


#*** Larger mutation effects

## We increase the standard deviation of mutational effects by 10 fold
plot(Normal(0,0.02))
plot!(Normal(0,0.2))

#--- Autarky
cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh_mut")

#--- Dyadic exchange
sigmasq_list = collect(0.5:0.5:8.5)
alpha_list = collect(0.1:0.05:0.9)
p = Progress(length(alpha_list) * length(sigmasq_list) , desc="Running simulations...")

for alpha_i in alpha_list
    for sigmasq_i in sigmasq_list
        parameters = Dict{Any,Any}(pairs((z_ini = 0., n_gen = 10000, n_ini = 2, n_patch = 2500, n_simul=10,
        boundaries=[-2.5,2.5],
        mu_m = 0.01, sigma_m = 0.2,
        de = "g",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
        o_x = -2., o_y = 2., sigma = sqrt(sigmasq_i), alpha = alpha_i, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
        write_file=true, parameters_to_omit=["z_ini","n_ini","sigma_m","j_print","str_selection","n_print","simplify","n_loci"],
        j_print = 1, n_print=10000)))
        evol_model(reproduction_WF,fitness_function,parameters)
        next!(p)
    end
end


#--- Market exchange
sigmasq_list = collect(0.5:0.5:8.5)
alpha_list = collect(0.1:0.05:0.9)
p = Progress(length(alpha_list) * length(sigmasq_list) , desc="Running simulations...")

for alpha_i in alpha_list
    for sigmasq_i in sigmasq_list
        parameters = Dict{Any,Any}(pairs((z_ini = 0., n_gen = 10000, n_ini = 5000, n_patch = 1, n_simul=10,
        boundaries=[-2.5,2.5],
        mu_m = 0.01, sigma_m = 0.2,
        de = "g",  exchange_fun = "market",name_model = "Exchpoly",
        o_x = -2., o_y = 2., sigma = sqrt(sigmasq_i), alpha = alpha_i, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
        write_file=true, parameters_to_omit=["z_ini","n_ini","sigma_m","j_print","str_selection","n_print","simplify"],
        j_print = 1, n_print=10000,simplify=false)))
        evol_model(reproduction_WF,fitness_function,parameters)
        next!(p)
    end
end


#*** Sexual reproduction
cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh_sexual")
#--- One Locus

## Dyadic bargaining

parameters = Dict{Any,Any}(pairs((z_ini = 2., n_gen = 15000, n_ini = 2, n_patch = 2500,
boundaries=[-10.,10.],
mu_m = 0.01, sigma_m = 0.02, n_loci = 1,str_selection=1.,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(1), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","n_print","simplify"],
j_print = 300,n_print=1)))

res=evol_model(reproduction_WF_sexual_multilocus,fitness_function,parameters)

## With replicates (for variance)

parameters = Dict{Any,Any}(pairs((z_ini = 2., n_gen = 10000, n_ini = 2, n_patch = 5000,n_simul=10,
boundaries=[-10.,10.],
mu_m = 0.01, sigma_m = 0.02, n_loci = 1,str_selection=1.,
de = "g",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(1), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","n_print","simplify"],
j_print = 250,n_print=1,distributed=false)))

res=evol_model(reproduction_WF_sexual_multilocus,fitness_function,parameters)

## Market
parameters = Dict{Any,Any}(pairs((z_ini = 2., n_gen = 25000, n_ini = 5000, n_patch = 1,
boundaries=[-10.,10],
mu_m = 0.01, sigma_m = 0.02, n_loci = 1,
de = "i",  exchange_fun = "market",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(5), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","n_print","simplify"],
j_print = 500,n_print=1,simplify=false)))

res=evol_model(reproduction_WF_sexual_multilocus,fitness_function,parameters)

## With replicates (for variance)

parameters = Dict{Any,Any}(pairs((z_ini = 2., n_gen = 10000, n_ini = 10000, n_patch = 1,n_simul=10,
boundaries=[-10.,10.],
mu_m = 0.01, sigma_m = 0.02, n_loci = 1,str_selection=1.,
de = "g",  exchange_fun = "market",name_model = "Exchpoly",
o_x = -2., o_y = 2., sigma = sqrt(1), alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y","var_z"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","n_print","simplify"],
j_print = 250,n_print=1,simplify=false)))

res=evol_model(reproduction_WF_sexual_multilocus,fitness_function,parameters)



#*** Two traits
cd(homedir()*"/OneDrive/Research/A1-Projects/2023_ExchPoly/Res/Simul/noh_2traits")

function constraint_on_traits(traits::Tuple)
    if sum(traits) < 1
        return(traits)
    else
        #-> They are considered to be effort dedicated to each.
        traits ./ sum(traits)
    end
end

#! Be careful, use o_x = 1 and o_y = 2 as integer to use the right production functions

#--- Dyadic exchange
parameters = Dict{Any,Any}(pairs((z_ini = (0.1,0.1), n_gen = 20000, n_ini = 2, n_patch = 500,
boundaries=[[0.,1.],[0.,1]],
mu_m = 0.01, sigma_m = 0.01,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = 1, o_y = 2, sigma = 5., alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify"],
j_print = 500,n_print=1)))

res=evol_model(reproduction_WF,fitness_function,parameters;genotype_to_phenotype_mapping=constraint_on_traits)

#--- Market exchange

parameters = Dict{Any,Any}(pairs((z_ini = (0.1,0.1), n_gen = 20000, n_ini = 1000, n_patch = 1,
boundaries=[[0.,1.],[0.,1]],
mu_m = 0.01, sigma_m = 0.01,
de = "i",  exchange_fun = "market",name_model = "Exchpoly",
o_x = 1, o_y = 2, sigma = 2., alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify"],
j_print = 500,n_print=1,simplify=false)))

res=evol_model(reproduction_WF,fitness_function,parameters;genotype_to_phenotype_mapping=constraint_on_traits)

#*** Two traits + sexual reproduction

#! Be careful, we change the definition of mapping here. 
function average_mapping(ind::Tuple{Vararg{<:AbstractMatrix{<:Real}}})
    constraint_on_traits(mean.(ind))
end


function constraint_on_traits(traits::Tuple)
    if sum(traits) < 1
        return(traits)
    else
        traits ./ sum(traits)
    end
end

#--- Dyadic exchange
parameters = Dict{Any,Any}(pairs((z_ini = (0.1,0.1), n_gen = 30000, n_ini = 2, n_patch = 2500,
boundaries=[[0.,1.],[0.,1]],
mu_m = 0.01, sigma_m = 0.01, n_loci = 1,
de = "i",  exchange_fun = "walrasian_bargain",name_model = "Exchpoly",
o_x = 1, o_y = 2, sigma = 5., alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify"],
j_print = 1500,n_print=1)))

res=evol_model(reproduction_WF_sexual_multilocus,fitness_function,parameters)


#--- Market
parameters = Dict{Any,Any}(pairs((z_ini = (0.1,0.1), n_gen = 25000, n_ini = 5000, n_patch = 1,
boundaries=[[0.,1.],[0.,1]],
mu_m = 0.01, sigma_m = 0.01, n_loci = 1,
de = "i",  exchange_fun = "market",name_model = "Exchpoly",
o_x = 1, o_y = 2, sigma = 5., alpha = 0.5, eta = 0, other_output_name=["h","q_x","q_y","c_x","c_y"],
write_file=true, parameters_to_omit=["z_ini","n_ini","mu_m","sigma_m","j_print","str_selection","n_print","simplify"],
j_print = 1000,n_print=1,simplify=false)))

res=evol_model(reproduction_WF_sexual_multilocus,fitness_function,parameters)
