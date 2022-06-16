vector<int> reverseLevelOrder(Node *root)
{
    // code here
    
    vector<int>ans;
    queue<Node*>nodes;
    nodes.push(root);
    while(!nodes.empty()){
        Node*front = nodes.front();
        nodes.pop();
        ans.push_back(front->data);
        if(front->right){
            nodes.push(front->right);
        }
        if(front->left){
            nodes.push(front->left);
        }
    }
    return ans;
}
# reverse level order traversal
vector<int> reverseLevelOrder(Node *root)
{
    // code here
    
    vector<int>ans;
    queue<Node*>nodes;
    nodes.push(root);
    while(!nodes.empty()){
        Node*front = nodes.front();
        nodes.pop();
        ans.push_back(front->data);
        if(front->right){
            nodes.push(front->right);
        }
        if(front->left){
            nodes.push(front->left);
        }
    }
    reverse(ans.begin(),ans.end());
    return ans;
}
