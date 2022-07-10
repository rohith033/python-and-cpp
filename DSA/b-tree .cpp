#include<bits/stdc++.h>
using namespace std;
// depth of tree - log(n)
//   each node has 3 pointers left right value
//   sub tree of x has x and its decendents 
//   depth --  no of node x is no of edges upto root node 
//   height -- no of edges in longest downward
//   in order traversal -- left sub tree-> root ->  right subtree

class node{
    public:
  int data;
  node* left;
  node* right;
};
node* creat_node(int a){
    node* temp = new node();
    temp->data = a;
    temp -> right =NULL;
    temp ->left = NULL;
    return temp;
}

  void inOrdertraversal(node* root){ 
  if(root==NULL) return;
 inOrdertraversal(root->left);
 cout<<root->data<<endl;
 inOrdertraversal(root->right);
 }
// post order traversal - left subtree right subtree root
void postordertraversal(node* root){
  if(root == NULL) return ;
  postordertraversal(root->left);
  postordertraversal(root->right);
  cout<<root->data<<endl;
}
// preorder traversal 
void preordertraversal(node* root){
  if(root == NULL) return;
  cout<<root->data<<endl;
  preordertraversal(root->left);
  preordertraversal(root->right);
}

int main(){
  node* first = creat_node(5);
//   first->data = 2;
  cout<<first->data<<" "<<first->left<<endl;
}
