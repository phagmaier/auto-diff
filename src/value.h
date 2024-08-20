#include <memory>
#include <unordered_set>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

enum Operation{
  None,
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Relu,
  Sig
};

class Value{
public:
  Value();
  Value(float val);
  Value(float val, std::shared_ptr<Value> &p1, Operation op);
  Value(float val, std::shared_ptr<Value> &p1, std::shared_ptr<Value> &p2, Operation op);


  float val;
  float grad;
  std::shared_ptr<Value> parent1;
  std::shared_ptr<Value> parent2;
  Operation op;
  std::unordered_set<std::shared_ptr<Value>> parents;
  
  friend std::shared_ptr<Value> operator+(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &rhs);
  friend std::shared_ptr<Value> operator*(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &rhs);

  friend std::shared_ptr<Value> pow(std::shared_ptr<Value> &val, float other);
  friend void build_topo(std::shared_ptr<Value> const &v, std::vector<std::shared_ptr<Value>> &topo, std::unordered_set<std::shared_ptr<Value>> &visited);
  friend void backward(std::shared_ptr<Value> &value);

  friend std::shared_ptr<Value> relu(std::shared_ptr<Value> &val);
  friend std::shared_ptr<Value> operator-(std::shared_ptr<Value> &val);
  friend std::shared_ptr<Value> operator-(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &rhs);
  friend std::shared_ptr<Value> operator/(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &rhs);
  friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& obj);
  friend std::shared_ptr<Value> PEM(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &middle, std::shared_ptr<Value> &rhs);
  void back_add(); 
  void back_mul();
  void back_pow();
  void back_relu();
  
  //depending on the op this function will call the appropriate grad func
  void get_grad_func();

};
