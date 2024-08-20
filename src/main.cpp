#include <iostream>
#include "value.h"
#include <memory>
#include <vector>
#include <random>
#include <cassert>

#define ALPHA 0.5

using PTR = std::shared_ptr<Value>;
using valVec = std::vector<std::vector<PTR>>;

float getRandomFloat() {
    // Static random engine and distribution
    static std::random_device rd;   // Random seed
    static std::mt19937 gen(rd());  // Mersenne Twister engine
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Range [0, 1]

    return dist(gen);
}

std::shared_ptr<Value> make_me(){
  Value v = Value();
  PTR ptr= std::make_shared<Value>(v);
  //return std::make_shared<Value>(69);
  return ptr;
}

std::shared_ptr<Value> make_ptr(float val){
  return std::make_shared<Value>(val);
}


std::vector<std::vector<PTR>>make_vec(int rows, int cols){
  valVec vec;
  for (int i=0;i<rows;++i){
    std::vector<PTR> row_vec;
    for (int x=0;x<cols;++x){
      row_vec.push_back(std::make_shared<Value>(getRandomFloat()));
    }
    vec.push_back(row_vec);
  }
  return vec;
}


valVec add_vecs(valVec &one, valVec &two){
  valVec vec;
  for (int i=0;i<one.size();++i){
    std::vector<PTR> row_vec;
    for (int x=0;x<one[0].size();++x){
      PTR aa = std::make_shared<Value>(13);
      PTR bb = one[i][x] + two[i][x];
      PTR cc = aa + bb;
      row_vec.push_back(cc);
    }
    vec.push_back(row_vec);
  }
  return vec;
}

valVec make_zeroes(int rows, int cols){
  valVec arr;
  for (int i=0;i<rows;++i){
    std::vector<std::shared_ptr<Value>> row;
    for (int x=0;x<cols;++x){
      row.push_back(std::make_shared<Value>());
    }
    arr.push_back(row);
  }
  return arr;
}

void apply_bias(valVec &vec,PTR &b1){
  for (int i=0;i<vec.size();++i){
    for (int x=0;x<vec[0].size();++x){
      vec[i][x] = vec[i][x] + b1;
    }
  }
}

void apply_activation(valVec &vec){
  for (int i=0;i<vec.size();++i){
    for (int x=0;x<vec[0].size();++x){
      vec[i][x] = relu(vec[i][x]);
    }
  }
}



valVec matMul(valVec &input, valVec &weights) {
  int rows_input = input.size();
  int cols_input = input[0].size();
  int rows_weights = weights.size();
  int cols_weights = weights[0].size();
  assert(cols_input == rows_weights);

  valVec results = make_zeroes(rows_input, cols_weights);
  
    // Perform matrix multiplication
  for (int i = 0; i < rows_input; ++i) {
    for (int j = 0; j < cols_weights; ++j) {
      for (int k = 0; k < cols_input; ++k) {
        //std::cout << "I J is: " << i << " " << j << "\n";
        //std::cout << results[i][j] << "\n";
        //std::cout << "ABOUT TO DO PEM\n";
        //std::cout << "AFTER: " << results[i][j] << "\n";
        //std::cout << "AFTER PEM\n";

        results[i][j] = PEM(results[i][j], input[i][k], weights[k][j]);
        //PTR temp = input[i][k] * weights[k][j];
        //results[i][j] = results[i][j] + temp;
      }
    }
  }

  return results;

}

valVec update_weights(valVec &w1){
  valVec new_weights;
  for (std::vector<std::shared_ptr<Value>> &i : w1){
    std::vector<std::shared_ptr<Value>> row;
    for (std::shared_ptr<Value> &x : i){
      row.push_back(make_ptr(x->val +x->grad * ALPHA));
    }
    new_weights.push_back(row);
  }
  return new_weights;
}

PTR get_error(valVec &results, valVec &targets){
  assert(results.size() == 1 && targets.size() == 1);
  assert(results[0].size() == targets[0].size());
  PTR total = make_ptr(0);
  for (int x = 0; x < results[0].size(); ++x){
    PTR temp = make_ptr(0.5);
    PTR diff = targets[0][x] - results[0][x];
    PTR squared_diff = pow(diff, 2);
    PTR error = temp * squared_diff;
    total = total + error;
  }
  return total;
}

void print(valVec &weights){
  for (std::vector<std::shared_ptr<Value>> &i : weights){
    for (PTR &x : i){
      std::cout << x << " ";
    }
    std::cout << "NEXT ROW\n";
  }
}

void print_grad(valVec &weights){
for (std::vector<std::shared_ptr<Value>> &i : weights){
  for (PTR &x : i){
    std::cout << x->grad << " ";
  }
  std::cout << "\n";
}
}




int main(){
  
  valVec targets= {
    {make_ptr(.01), make_ptr(.99)}
  };

  valVec w1 = {
    {make_ptr(.15), make_ptr(.2)},
    {make_ptr(.25), make_ptr(.3)}
  };

  valVec w2 = {
    {make_ptr(.4),make_ptr(.45)},
    {make_ptr(.5), make_ptr(.55)}
  };

  valVec inputs = {
    {make_ptr(0.05),make_ptr(.1)},
  };

  PTR b1 = make_ptr(.35);
  PTR b2 = make_ptr(.6);
  valVec h1 = matMul(inputs,w1); 
  apply_bias(h1,b1);
  apply_activation(h1);
  valVec output = matMul(h1,w2);
  apply_bias(output,b2);
  apply_activation(output);
  std::cout << "THE OUTPUT I GOT WAS: \n";

  print(output);

  
  PTR err = get_error(output,targets);
  std::cout << "The error is: " << err << "\n";
  backward(err);
  std::cout << "GRADIENT w1\n";
  print_grad(w1);
  std::cout << "GRADIENT w2\n";
  print_grad(w2);
  valVec w1_new = update_weights(w1);
  valVec w2_new = update_weights(w2);
  std::cout << "Weights 1 post update: \n";
  print(w1_new);

  std::cout << "Weights 2 post update: \n";
  print(w2_new);

  std::cout << "\n\n\n\n\n\n";

  /*

  PTR a = make_ptr(5);
  PTR b = make_ptr(6);
  PTR c = make_ptr(7);
  PTR step1 = pow(a,2);
  PTR step2 = pow(b,2);
  PTR step3 = step1 - step2;
  PTR result =  step3 - c;
  backward(result);
  std::cout << a->grad << "\n";
  std::cout << b->grad << "\n";
  std::cout << c->grad << "\n";
  std::cout << a << "\n";
  std::cout << b << "\n";
  std::cout << c << "\n";
  */

  /*
  PTR a = make_ptr(5);
  PTR b = make_ptr(6);
  PTR c = make_ptr(7);
  PTR temp1 = a * b;
  PTR temp2 = temp1 * c;
  backward(temp2);
  std::cout << a->grad << "\n";
  std::cout << b->grad << "\n";
  std::cout << c->grad << "\n";
  std::cout << temp2->grad << "\n";
*/
  /*
  PTR a = make_ptr(2);
  PTR b = make_ptr(3);
  PTR c = make_ptr(4);
  PTR d = PEM(c,a,b);
  PTR post_relu = relu(d);
  backward(post_relu);
  std::cout << a;
  std::cout << b;
  std::cout << c;
  std::cout << d;
  std::cout << post_relu;
  */  



  

  return 0;
}
