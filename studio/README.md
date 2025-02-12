# AgentFlow Studio

A modern web interface for managing and monitoring AgentFlow workflows.

## Latest Version

Current version: v0.1.1
- Fixed workflow transform functions to handle step and context parameters
- Added feature engineering and outlier removal transforms
- Improved test suite and type hints
- Enhanced error handling and validation

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

## Features

- **Workflow Visualization**
  - Interactive workflow graph
  - Step dependency visualization
  - Real-time status updates

- **Performance Monitoring**
  - Step execution times
  - Memory usage tracking
  - Error monitoring

- **Configuration Management**
  - Visual workflow configuration
  - Agent configuration editor
  - Transform function management

## Development

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm start
```

3. Run tests:
```bash
npm test
```

4. Build for production:
```bash
npm run build
```

## Project Structure

```
studio/
├── src/
│   ├── components/     # React components
│   ├── hooks/         # Custom hooks
│   ├── pages/         # Page components
│   ├── services/      # API services
│   └── utils/         # Utility functions
├── public/            # Static assets
└── tests/            # Test files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Implement your changes
5. Run the test suite
6. Create a Pull Request

## Learn More

- [React Documentation](https://reactjs.org/)
- [AgentFlow Documentation](../docs/README.md)
- [Testing Guide](../docs/testing.md)
